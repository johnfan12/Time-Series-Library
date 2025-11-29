"""
PatchTST with End-to-End Learned LoMoE using Time Series Statistical Features for Routing.

This model (E2E_F = E2E with Features) uses statistical features extracted from the 
raw input time series as router input, instead of encoder output. This makes the 
routing decision based on explicit time series characteristics like:
- Mean, std, min, max, median
- Skewness, kurtosis
- Trend slope
- FFT energy features
- Autocorrelation features

The intuition is that different time series patterns (trending, seasonal, noisy, etc.)
may benefit from different expert specializations.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.PatchTST import FlattenHead, Model as PatchTSTBase
from utils.ts_stats import FeatureExtractionConfig, extract_ts_features


class LoRAExpert(nn.Module):
    """Low-rank adapter used as a lightweight expert."""

    def __init__(self, d_in: int, d_out: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        temp = nn.functional.linear(x_flat, self.lora_A)
        delta = nn.functional.linear(temp, self.lora_B)
        delta = delta * self.scaling
        return delta.view(*orig_shape[:-1], -1)


class TSFeatureExtractor(nn.Module):
    """
    Extracts statistical features from raw time series for routing.
    
    This is a non-parametric feature extraction that runs on CPU/numpy,
    then converts to tensor for the router network.
    """
    
    def __init__(
        self,
        max_acf_lag: int = 6,
        top_k_fft: int = 3,
        poly_order: int = 1,
        clip_value: float = 5.0,
    ):
        super().__init__()
        self.config = FeatureExtractionConfig(
            max_acf_lag=max_acf_lag,
            top_k_fft=top_k_fft,
            poly_order=poly_order,
            clip_value=clip_value,
            time_axis=0,  # time is first axis in our input
        )
        # Calculate feature dimension
        # Per variable: 10 base features + max_acf_lag ACF features
        # Then we concatenate mean and std across variables
        per_var_dim = 10 + max_acf_lag
        self.feature_dim = per_var_dim * 2  # mean + std aggregation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical features from batch of time series.
        
        Args:
            x: [B, seq_len, n_vars] input time series
            
        Returns:
            features: [B, feature_dim] statistical features
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Move to CPU and convert to numpy for feature extraction
        x_np = x.detach().cpu().numpy()
        
        features_list = []
        for i in range(batch_size):
            # x_np[i] shape: [seq_len, n_vars]
            feat = extract_ts_features(x_np[i], self.config)
            features_list.append(feat)
        
        # Stack and convert back to tensor
        features = np.stack(features_list, axis=0)
        features = torch.from_numpy(features).float().to(device)
        
        return features


class FeatureBasedRouter(nn.Module):
    """
    Router that uses time series statistical features for expert selection.
    
    Instead of using encoder output, this router takes pre-computed statistical
    features (trend, periodicity, dispersion, etc.) to make routing decisions.
    """

    def __init__(
        self,
        feature_dim: int,
        num_experts: int,
        top_k: int = 2,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.temperature = temperature
        self.noise_std = noise_std
        
        hidden_dim = hidden_dim or max(64, feature_dim)
        self.router_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def forward(
        self, 
        features: torch.Tensor,
        add_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, feature_dim] statistical features
            add_noise: Whether to add noise for load balancing
            
        Returns:
            routing_weights: [B, top_k] normalized weights
            selected_indices: [B, top_k] selected expert indices
            router_probs: [B, num_experts] full probability distribution
        """
        logits = self.router_net(features)
        
        if add_noise and self.noise_std > 0 and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        probs = torch.softmax(logits / self.temperature, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        denom = torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-6)
        topk_weights = topk_weights / denom
        
        return topk_weights, topk_indices, probs


class LoMoEOutputHeadE2E_F(nn.Module):
    """
    LoMoE Output Head with Feature-based Router.
    
    Uses statistical features extracted from raw input time series
    for routing decisions, rather than encoder output.
    """

    def __init__(
        self,
        base_head: nn.Module,
        feature_dim: int,
        num_experts: int = 4,
        rank: int = 8,
        top_k: int = 2,
        router_hidden_dim: Optional[int] = None,
        router_temperature: float = 1.0,
        router_noise_std: float = 0.1,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.base_head = base_head
        self.n_vars = base_head.n_vars
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        in_features = base_head.linear.in_features
        out_features = base_head.linear.out_features
        
        # Feature-based router
        self.router = FeatureBasedRouter(
            feature_dim=feature_dim,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=router_hidden_dim,
            temperature=router_temperature,
            noise_std=router_noise_std,
        )
        
        # LoRA experts
        self.experts = nn.ModuleList([
            LoRAExpert(in_features, out_features, rank=rank, alpha=lora_alpha)
            for _ in range(num_experts)
        ])
        
        self._forced_expert_idx: Optional[int] = None

    def set_single_expert_mode(self, expert_idx: Optional[int]) -> None:
        if expert_idx is None:
            self._forced_expert_idx = None
            return
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(f"expert_idx must be in [0, {self.num_experts})")
        self._forced_expert_idx = expert_idx

    def replicate_expert(self, src_idx: int = 0) -> None:
        if src_idx < 0 or src_idx >= self.num_experts:
            raise ValueError(f"src_idx must be in [0, {self.num_experts})")
        src_state = {
            name: param.detach().clone()
            for name, param in self.experts[src_idx].state_dict().items()
        }
        for idx, expert in enumerate(self.experts):
            if idx != src_idx:
                expert.load_state_dict(src_state)

    def lora_parameters(self):
        for expert in self.experts:
            yield from expert.parameters()

    def router_parameters(self):
        yield from self.router.parameters()

    def forward(
        self,
        x: torch.Tensor,
        ts_features: torch.Tensor,
        router_override: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, n_vars, d_model, patch_num] encoder output
            ts_features: [B, feature_dim] statistical features from raw input
            router_override: Optional tuple to override router
            
        Returns:
            output: [B, n_vars, pred_len] predictions
            router_probs: [B, num_experts] routing probabilities
        """
        batch_size = x.shape[0]
        n_vars = x.shape[1]
        device = x.device
        
        # Base head forward
        flattened = self.base_head.flatten(x)
        base_out = self.base_head.linear(flattened)
        base_out = self.base_head.dropout(base_out)
        
        # Routing based on time series features
        if router_override is not None:
            routing_weights, selected_indices, router_probs = router_override
        elif self._forced_expert_idx is not None:
            routing_weights = torch.ones(batch_size, 1, device=device)
            selected_indices = torch.full(
                (batch_size, 1), self._forced_expert_idx, dtype=torch.long, device=device
            )
            router_probs = torch.zeros(batch_size, self.num_experts, device=device)
            router_probs[:, self._forced_expert_idx] = 1.0
        else:
            routing_weights, selected_indices, router_probs = self.router(
                ts_features, add_noise=self.training
            )
        
        # Compute expert outputs
        flat_2d = flattened.reshape(batch_size * n_vars, -1)
        expert_outputs = [
            expert(flat_2d).view(batch_size, n_vars, -1) 
            for expert in self.experts
        ]
        
        # Weighted combination
        moe_delta = torch.zeros_like(base_out)
        for b in range(batch_size):
            for k in range(selected_indices.shape[1]):
                expert_idx = selected_indices[b, k].item()
                weight = routing_weights[b, k]
                moe_delta[b] = moe_delta[b] + weight * expert_outputs[expert_idx][b]
        
        final_out = base_out + moe_delta
        return final_out, router_probs


class Model(PatchTSTBase):
    """
    PatchTST with End-to-End LoMoE using Time Series Statistical Features for Routing.
    
    Key differences from PatchTST_LoMoE_E2E:
    - Router input: statistical features from raw input (not encoder output)
    - Features include: mean, std, trend, periodicity, autocorrelation, etc.
    - More interpretable routing based on explicit time series characteristics
    
    The intuition is that different types of time series (trending, seasonal, 
    stationary, volatile) may benefit from different expert specializations.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(configs, patch_len=patch_len, stride=stride)
        
        self._router_probs = None
        self._lomoe_active = self.task_name in {
            'long_term_forecast', 'short_term_forecast', 'multi_dataset_forecast'
        }
        self._backbone_frozen = False
        
        # Feature extraction config
        max_acf_lag = getattr(configs, 'router_feat_max_acf', 6)
        top_k_fft = getattr(configs, 'router_feat_topk_fft', 3)
        poly_order = getattr(configs, 'router_feat_poly_order', 1)
        clip_value = getattr(configs, 'router_feat_clip', 5.0)
        
        # Initialize feature extractor
        self.feature_extractor = TSFeatureExtractor(
            max_acf_lag=max_acf_lag,
            top_k_fft=top_k_fft,
            poly_order=poly_order,
            clip_value=clip_value,
        )
        
        if self._lomoe_active:
            if not isinstance(self.head, FlattenHead):
                raise ValueError('LoMoE head expects a FlattenHead for forecasting tasks.')
            
            num_experts = getattr(configs, 'num_experts', 4)
            rank = getattr(configs, 'lora_rank', 8)
            top_k = getattr(configs, 'moe_topk', 2)
            router_hidden_dim = getattr(configs, 'router_hidden_dim', None)
            router_temperature = getattr(configs, 'router_temperature', 1.0)
            router_noise_std = getattr(configs, 'router_noise_std', 0.1)
            lora_alpha = getattr(configs, 'lora_alpha', 16.0)
            
            self.head = LoMoEOutputHeadE2E_F(
                base_head=self.head,
                feature_dim=self.feature_extractor.feature_dim,
                num_experts=num_experts,
                rank=rank,
                top_k=top_k,
                router_hidden_dim=router_hidden_dim,
                router_temperature=router_temperature,
                router_noise_std=router_noise_std,
                lora_alpha=lora_alpha,
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Extract statistical features from raw input BEFORE normalization
        # x_enc: [B, seq_len, n_vars]
        ts_features = self.feature_extractor(x_enc)
        
        # Instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # Encoder
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Head with LoMoE (pass ts_features for routing)
        if self._lomoe_active:
            head_out, router_probs = self.head(enc_out, ts_features)
            self._router_probs = router_probs
        else:
            head_out = self.head(enc_out)
            self._router_probs = None
        
        dec_out = head_out.permute(0, 2, 1)

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {'long_term_forecast', 'short_term_forecast', 'multi_dataset_forecast'}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            dec_out = dec_out[:, -self.pred_len:, :]
            if self._lomoe_active:
                return dec_out, self._router_probs
            return dec_out
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    def get_router_probs(self):
        return self._router_probs

    # ---- LoMoE training utilities ----
    
    def set_single_expert_mode(self, expert_idx: Optional[int]) -> None:
        if isinstance(self.head, LoMoEOutputHeadE2E_F):
            self.head.set_single_expert_mode(expert_idx)

    def replicate_primary_expert(self, src_idx: int = 0) -> None:
        if isinstance(self.head, LoMoEOutputHeadE2E_F):
            self.head.replicate_expert(src_idx)

    def set_cluster_router_enabled(self, enabled: bool) -> None:
        """Compatibility method (no-op)."""
        pass

    def _iter_lora_parameters(self):
        if isinstance(self.head, LoMoEOutputHeadE2E_F):
            yield from self.head.lora_parameters()

    def _iter_router_parameters(self):
        if isinstance(self.head, LoMoEOutputHeadE2E_F):
            yield from self.head.router_parameters()

    def freeze_backbone_for_lora(self) -> None:
        if self._backbone_frozen:
            return
        lora_params = set(self._iter_lora_parameters())
        router_params = set(self._iter_router_parameters())
        trainable_params = lora_params | router_params
        
        if not trainable_params:
            raise RuntimeError("LoMoE head not found; cannot freeze backbone.")
        
        for param in self.parameters():
            param.requires_grad = False
        for param in trainable_params:
            param.requires_grad = True
        
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        if not self._backbone_frozen:
            return
        for param in self.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
    
    def get_expert_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        if router_probs is None:
            return torch.tensor(0.0)
        
        expert_usage = router_probs.mean(dim=0)
        num_experts = expert_usage.shape[0]
        uniform = torch.full_like(expert_usage, 1.0 / num_experts)
        load_balance_loss = ((expert_usage - uniform) ** 2).sum() * num_experts
        
        return load_balance_loss
