from typing import Optional

import torch
import torch.nn as nn

from .HyperTS import TransformerBackbone
from .hyper_feature_bank import SimpleTSFeature


class Model(nn.Module):
    """Transformer backbone with feature-conditioned MLP head."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.c_out = configs.c_out

        if self.task_name in ["classification", "anomaly_detection", "imputation"]:
            self.pred_len = self.seq_len
        else:
            self.pred_len = configs.pred_len

        d_model = configs.d_model
        feat_dim = getattr(configs, "feature_head_dim", d_model)
        feat_freqs = getattr(configs, "feature_n_freqs", 3)
        feat_ar_order = getattr(configs, "feature_ar_order", 3)
        feat_ar_reg = getattr(configs, "feature_ar_reg", 1e-4)
        feat_arima_max_p = getattr(configs, "feature_arima_max_p", 3)
        feat_arima_max_d = getattr(configs, "feature_arima_max_d", 2)
        feat_methods = getattr(configs, "feature_methods", None)

        self.backbone = TransformerBackbone(configs)
        self.feat_extractor = SimpleTSFeature(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_feat=feat_dim,
            method_names=feat_methods,
            n_freqs=feat_freqs,
            ar_order=feat_ar_order,
            ar_regularizer=feat_ar_reg,
            arima_max_p=feat_arima_max_p,
            arima_max_d=feat_arima_max_d,
        )

        fusion_dim = d_model + feat_dim
        head_hidden = getattr(configs, "feature_head_hidden", fusion_dim * 2)
        head_dropout = getattr(configs, "feature_head_dropout", getattr(configs, "dropout", 0.1))

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, self.c_out),
        )

    def _forward_core(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        backbone_out = self.backbone(x_enc, x_mark_enc)
        feat = self.feat_extractor(x_enc)
        feat_seq = feat.unsqueeze(1).expand(-1, backbone_out.size(1), -1)
        fused = torch.cat([backbone_out, feat_seq], dim=-1)
        fused = self.fusion_norm(fused)
        return self.head(fused)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: Optional[torch.Tensor],
        x_mark_dec: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self._forward_core(x_enc, x_mark_enc)
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return outputs[:, -self.pred_len:, :]
        if self.task_name in ["imputation", "anomaly_detection"]:
            return outputs
        raise NotImplementedError(f"Task {self.task_name} is not supported by FeatureTS.")
