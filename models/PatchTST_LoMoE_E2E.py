"""
PatchTST with End-to-End Learned LoMoE (Mixture of LoRA Experts).

This model uses a neural network router trained end-to-end, without requiring
pre-computed clustering artifacts. The router learns to route samples to
different LoRA experts based on the input time series characteristics.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.PatchTST import FlattenHead, Model as PatchTSTBase


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
        # x: [..., d_in] -> [..., d_out]
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        temp = nn.functional.linear(x_flat, self.lora_A)  # [*, rank]
        delta = nn.functional.linear(temp, self.lora_B)   # [*, d_out]
        delta = delta * self.scaling
        return delta.view(*orig_shape[:-1], -1)


class LearnedRouter(nn.Module):
    """
    Neural network router that learns to select top-k experts.
    
    Uses a two-layer MLP with temporal pooling to compute routing probabilities.
    """

    def __init__(
        self,
        d_model: int,
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
        self.noise_std = noise_std  # For load balancing during training
        
        hidden_dim = hidden_dim or max(64, d_model // 2)
        self.router_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self, 
        pooled_feat: torch.Tensor,
        add_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled_feat: [B, d_model] pooled sequence features
            add_noise: Whether to add noise for load balancing (training only)
            
        Returns:
            routing_weights: [B, top_k] normalized weights for selected experts
            selected_indices: [B, top_k] indices of selected experts
            router_probs: [B, num_experts] full probability distribution
        """
        logits = self.router_net(pooled_feat)  # [B, num_experts]
        
        # Add noise during training for better load balancing
        if add_noise and self.noise_std > 0 and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Temperature-scaled softmax
        probs = torch.softmax(logits / self.temperature, dim=-1)
        
        # Top-K selection
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize selected weights to sum to 1
        denom = torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-6)
        topk_weights = topk_weights / denom
        
        return topk_weights, topk_indices, probs


class LoMoEOutputHeadE2E(nn.Module):
    """
    End-to-End LoMoE Output Head with learned router.
    
    Wraps an existing FlattenHead with LoRA experts and a neural network router.
    The router is trained end-to-end with the model.
    """

    def __init__(
        self,
        base_head: nn.Module,
        d_model: int,
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
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        in_features = base_head.linear.in_features
        out_features = base_head.linear.out_features
        
        # Learned router
        self.router = LearnedRouter(
            d_model=d_model,
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
        
        # For single expert mode (warmup training)
        self._forced_expert_idx: Optional[int] = None

    def set_single_expert_mode(self, expert_idx: Optional[int]) -> None:
        """Force all samples to use a single expert (for warmup training)."""
        if expert_idx is None:
            self._forced_expert_idx = None
            return
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(f"expert_idx must be in [0, {self.num_experts})")
        self._forced_expert_idx = expert_idx

    def replicate_expert(self, src_idx: int = 0) -> None:
        """Copy weights from source expert to all other experts."""
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
        """Iterate over all LoRA expert parameters."""
        for expert in self.experts:
            yield from expert.parameters()

    def router_parameters(self):
        """Iterate over router parameters."""
        yield from self.router.parameters()

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool encoder output to get sequence-level features for routing.
        x: [B, n_vars, d_model, patch_num]
        Returns: [B, d_model]
        """
        # Average over variables and patches
        pooled = x.mean(dim=1)  # [B, d_model, patch_num]
        pooled = pooled.mean(dim=-1)  # [B, d_model]
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        router_override: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, n_vars, d_model, patch_num] encoder output
            router_override: Optional tuple to override router (for compatibility)
            
        Returns:
            output: [B, n_vars, pred_len] predictions
            router_probs: [B, num_experts] routing probabilities
        """
        batch_size = x.shape[0]
        n_vars = x.shape[1]
        device = x.device
        
        # Base head forward pass
        flattened = self.base_head.flatten(x)  # [B, n_vars, in_features]
        base_out = self.base_head.linear(flattened)  # [B, n_vars, out_features]
        base_out = self.base_head.dropout(base_out)
        
        # Routing
        if router_override is not None:
            routing_weights, selected_indices, router_probs = router_override
        elif self._forced_expert_idx is not None:
            # Single expert mode (warmup)
            routing_weights = torch.ones(batch_size, 1, device=device)
            selected_indices = torch.full(
                (batch_size, 1), self._forced_expert_idx, dtype=torch.long, device=device
            )
            router_probs = torch.zeros(batch_size, self.num_experts, device=device)
            router_probs[:, self._forced_expert_idx] = 1.0
        else:
            # Normal routing with learned router
            pooled_feat = self._pool_features(x)
            routing_weights, selected_indices, router_probs = self.router(
                pooled_feat, add_noise=self.training
            )
        
        # Compute all expert outputs (for simplicity; can be optimized)
        flat_2d = flattened.reshape(batch_size * n_vars, -1)  # [B*n_vars, in_features]
        expert_outputs = [
            expert(flat_2d).view(batch_size, n_vars, -1) 
            for expert in self.experts
        ]  # List of [B, n_vars, out_features]
        
        # Weighted combination of selected experts
        moe_delta = torch.zeros_like(base_out)
        for b in range(batch_size):
            for k in range(selected_indices.shape[1]):
                expert_idx = selected_indices[b, k].item()
                weight = routing_weights[b, k]
                moe_delta[b] = moe_delta[b] + weight * expert_outputs[expert_idx][b]
        
        # Final output = base + MoE delta
        final_out = base_out + moe_delta
        
        return final_out, router_probs


class Model(PatchTSTBase):
    """
    PatchTST with End-to-End Learned LoMoE.
    
    This model uses a neural network router that is trained end-to-end,
    without requiring pre-computed clustering artifacts.
    
    Key differences from PatchTST_LoMoE:
    - Router is a learned MLP, not based on pre-computed clusters
    - Router is trained jointly with the model
    - Supports auxiliary load balancing loss
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(configs, patch_len=patch_len, stride=stride)
        
        self._router_probs = None
        self._lomoe_active = self.task_name in {
            'long_term_forecast', 'short_term_forecast', 'multi_dataset_forecast'
        }
        self._backbone_frozen = False
        
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
            
            self.head = LoMoEOutputHeadE2E(
                base_head=self.head,
                d_model=configs.d_model,
                num_experts=num_experts,
                rank=rank,
                top_k=top_k,
                router_hidden_dim=router_hidden_dim,
                router_temperature=router_temperature,
                router_noise_std=router_noise_std,
                lora_alpha=lora_alpha,
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
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

        # Head with LoMoE
        if self._lomoe_active:
            head_out, router_probs = self.head(enc_out)
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
        """Get the last computed router probabilities."""
        return self._router_probs

    # ---- LoMoE training utilities ----
    
    def set_single_expert_mode(self, expert_idx: Optional[int]) -> None:
        """Force all samples to use a single expert (for warmup training)."""
        if isinstance(self.head, LoMoEOutputHeadE2E):
            self.head.set_single_expert_mode(expert_idx)

    def replicate_primary_expert(self, src_idx: int = 0) -> None:
        """Copy weights from source expert to all other experts."""
        if isinstance(self.head, LoMoEOutputHeadE2E):
            self.head.replicate_expert(src_idx)

    def set_cluster_router_enabled(self, enabled: bool) -> None:
        """Compatibility method (no-op for E2E model since there's no cluster router)."""
        pass

    def _iter_lora_parameters(self):
        """Iterate over all LoRA expert parameters."""
        if isinstance(self.head, LoMoEOutputHeadE2E):
            yield from self.head.lora_parameters()

    def _iter_router_parameters(self):
        """Iterate over router parameters."""
        if isinstance(self.head, LoMoEOutputHeadE2E):
            yield from self.head.router_parameters()

    def freeze_backbone_for_lora(self) -> None:
        """Freeze backbone, keep only LoRA experts and router trainable."""
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
        """Unfreeze all parameters."""
        if not self._backbone_frozen:
            return
        for param in self.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
    
    def get_expert_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss to encourage uniform expert usage.
        
        Args:
            router_probs: [B, num_experts] routing probabilities
            
        Returns:
            Scalar loss encouraging uniform distribution across experts
        """
        if router_probs is None:
            return torch.tensor(0.0)
        
        # Mean probability per expert across batch
        expert_usage = router_probs.mean(dim=0)  # [num_experts]
        num_experts = expert_usage.shape[0]
        
        # Target uniform distribution
        uniform = torch.full_like(expert_usage, 1.0 / num_experts)
        
        # Squared deviation from uniform (coefficient of variation based loss)
        load_balance_loss = ((expert_usage - uniform) ** 2).sum() * num_experts
        
        return load_balance_loss
