import math
from typing import Optional, Tuple

import torch
from torch import nn


class LoRALayer(nn.Module):
    """Low-rank adapter used as a lightweight expert."""

    def __init__(self, d_in: int, d_out: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        temp = nn.functional.linear(x_flat, self.lora_A)
        delta = nn.functional.linear(temp, self.lora_B) * self.scaling
        return delta.view(*orig_shape[:-1], -1)


class TopKRouter(nn.Module):
    """Simple router that pools temporal features and selects top-k experts."""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, hidden_dim: Optional[int] = None):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        hidden_dim = hidden_dim or max(1, d_model // 2)
        self.router_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pooled_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router_net(pooled_feat)
        probs = self.softmax(logits)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        denom = torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-6)
        topk_weights = topk_weights / denom
        return topk_weights, topk_indices, probs


class LoMoEOutputHead(nn.Module):
    """Wraps an existing FlattenHead with a LoMoE adapter block."""

    def __init__(
        self,
        base_head: nn.Module,
        d_model: int,
        num_experts: int = 4,
        rank: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        self.n_vars = base_head.n_vars
        in_features = base_head.linear.in_features
        out_features = base_head.linear.out_features
        self.router = TopKRouter(d_model, num_experts, top_k=top_k)
        self.experts = nn.ModuleList([
            LoRALayer(in_features, out_features, rank=rank) for _ in range(num_experts)
        ])

    def _pool_router_features(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)  # [B, d_model, patch_num]
        pooled = pooled.mean(dim=-1)  # [B, d_model]
        return pooled

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Base path
        flattened = self.base_head.flatten(x)
        base_out = self.base_head.linear(flattened)
        base_out = self.base_head.dropout(base_out)

        # Router
        router_inputs = self._pool_router_features(x)
        routing_weights, selected_indices, router_probs = self.router(router_inputs)

        batch_size = flattened.shape[0]
        flat_2d = flattened.view(batch_size * self.n_vars, -1)
        expert_outputs = [expert(flat_2d).view(batch_size, self.n_vars, -1) for expert in self.experts]

        moe_delta = torch.zeros_like(base_out)
        for b in range(batch_size):
            sample_delta = torch.zeros_like(base_out[b])
            for pos, expert_idx in enumerate(selected_indices[b]):
                weight = routing_weights[b, pos]
                idx = int(expert_idx.item())
                sample_delta = sample_delta + weight * expert_outputs[idx][b]
            moe_delta[b] = sample_delta

        final_out = base_out + moe_delta
        return final_out, router_probs
