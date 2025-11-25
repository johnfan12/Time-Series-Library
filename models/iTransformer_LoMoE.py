import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from layers.ClusterRouter import ClusterRouterArtifacts, StatsClusterRouter
from layers.LoMoE_Layers import LoMoELinearHead
from models.iTransformer import Model as ITransformerBase
from utils.ts_stats import FeatureExtractionConfig


class Model(ITransformerBase):
    """iTransformer variant equipped with a lightweight MoE projection head."""

    def __init__(self, configs):
        super().__init__(configs)
        self._router_probs: Optional[torch.Tensor] = None
        self._router_override: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._lomoe_active = self.task_name in {"long_term_forecast", "short_term_forecast"}
        self._cluster_router_enabled = True
        self._backbone_frozen = False
        self.cluster_router: Optional[StatsClusterRouter] = None

        if self._lomoe_active:
            if not isinstance(self.projection, nn.Linear):
                raise ValueError("iTransformer LoMoE expects a Linear projection head for forecasting tasks.")
            num_experts = getattr(configs, "num_experts", 4)
            rank = getattr(configs, "lora_rank", 8)
            top_k = getattr(configs, "moe_topk", 2)
            self.projection = LoMoELinearHead(
                base_linear=self.projection,
                d_model=configs.d_model,
                num_experts=num_experts,
                rank=rank,
                top_k=top_k,
            )
            cluster_dir = getattr(configs, "cluster_artifact_dir", None)
            if cluster_dir:
                reducer_path = Path(cluster_dir) / "reducer.joblib"
                cluster_path = Path(cluster_dir) / "cluster.joblib"
                if not reducer_path.exists() or not cluster_path.exists():
                    raise FileNotFoundError(
                        f"Cluster artifacts not found under {cluster_dir}: expected reducer.joblib and cluster.joblib"
                    )
                feature_cfg_path = Path(cluster_dir) / "feature_cfg.json"
                if feature_cfg_path.exists():
                    with open(feature_cfg_path, "r", encoding="utf-8") as f:
                        cfg_dict = json.load(f)
                    feature_cfg = FeatureExtractionConfig(**cfg_dict)
                else:
                    feature_cfg = FeatureExtractionConfig(
                        max_acf_lag=getattr(configs, "cluster_feat_max_acf", 6),
                        top_k_fft=getattr(configs, "cluster_feat_topk_fft", 3),
                        poly_order=getattr(configs, "cluster_feat_poly_order", 1),
                        clip_value=getattr(configs, "cluster_feat_clip", None),
                    )
                device = torch.device(getattr(configs, "device", "cpu")) if hasattr(configs, "device") else None
                artifacts = ClusterRouterArtifacts(
                    reducer_path=reducer_path,
                    cluster_path=cluster_path,
                    feature_cfg=feature_cfg,
                )
                self.cluster_router = StatsClusterRouter(
                    artifacts=artifacts,
                    top_k=top_k,
                    temperature=getattr(configs, "cluster_router_temperature", 1.0),
                    metric=getattr(configs, "cluster_router_metric", "euclidean"),
                    device=device,
                )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        stats_source = x_enc.detach() if self.cluster_router else None
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, n_vars = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        if self._lomoe_active:
            router_override = None
            if (
                self.cluster_router is not None
                and stats_source is not None
                and self._cluster_router_enabled
            ):
                router_override = self.cluster_router.route(stats_source)
            proj_out, router_probs = self.projection(enc_out, router_override=router_override)
            self._router_probs = router_probs
            self._router_override = router_override
        else:
            proj_out = self.projection(enc_out)
            self._router_probs = None
            self._router_override = None

        dec_out = proj_out.permute(0, 2, 1)[:, :, :n_vars]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            dec_out = dec_out[:, -self.pred_len:, :]
            if self._lomoe_active:
                return dec_out, self._router_probs
            return dec_out
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    def get_router_probs(self):
        return self._router_probs

    def get_router_override(self):
        return self._router_override

    def set_cluster_router_enabled(self, enabled: bool) -> None:
        self._cluster_router_enabled = bool(enabled)

    def set_single_expert_mode(self, expert_idx: Optional[int]) -> None:
        if isinstance(self.projection, LoMoELinearHead):
            self.projection.set_single_expert_mode(expert_idx)

    def replicate_primary_expert(self, src_idx: int = 0) -> None:
        if isinstance(self.projection, LoMoELinearHead):
            self.projection.replicate_expert(src_idx)

    def _iter_lora_parameters(self):
        if isinstance(self.projection, LoMoELinearHead):
            yield from self.projection.lora_parameters()

    def freeze_backbone_for_lora(self) -> None:
        if self._backbone_frozen:
            return
        lora_params = list(self._iter_lora_parameters())
        if not lora_params:
            raise RuntimeError("LoMoE head not found; cannot freeze backbone for LoRA-only training.")
        for param in self.parameters():
            param.requires_grad = False
        for param in lora_params:
            param.requires_grad = True
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        if not self._backbone_frozen:
            return
        for param in self.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
