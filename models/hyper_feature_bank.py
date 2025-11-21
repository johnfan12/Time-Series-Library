"""Feature extraction utilities for HyperTS-style models."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class _BaseFeature(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class StatsFeature(_BaseFeature):
    """Basic statistics: mean, std, last value, diff mean, autocorrelation."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = in_channels * 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        var = x.var(dim=1, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        last = x[:, -1, :]
        if x.shape[1] > 1:
            diff = x[:, 1:, :] - x[:, :-1, :]
            diff_mean = diff.mean(dim=1)
            autocorr = (x[:, 1:, :] * x[:, :-1, :]).mean(dim=1)
        else:
            diff_mean = torch.zeros_like(mean)
            autocorr = torch.zeros_like(mean)
        return torch.cat([mean, std, last, diff_mean, autocorr], dim=-1)


class TrendSeasonFeature(_BaseFeature):
    """Trend slope + dominant seasonal frequency magnitudes."""

    def __init__(self, seq_len: int, in_channels: int, n_freqs: int = 3) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.n_freqs = max(1, n_freqs)
        self.output_dim = in_channels * (1 + self.n_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        centered = x - mean.unsqueeze(1)
        time_idx = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
        time_idx = time_idx - time_idx.mean()
        slope_num = (centered * time_idx.view(1, -1, 1)).sum(dim=1)
        slope_den = torch.clamp(time_idx.pow(2).sum(), min=1e-6)
        slope = slope_num / slope_den

        freq_domain = torch.fft.rfft(centered.permute(0, 2, 1), dim=-1)
        amp = freq_domain.abs()
        if amp.shape[-1] > 1:
            seasonal_amp = amp[..., 1:]
            topk = min(self.n_freqs, seasonal_amp.shape[-1])
            seasonal_topk, _ = torch.topk(seasonal_amp, k=topk, dim=-1)
            if topk < self.n_freqs:
                pad = torch.zeros(
                    *seasonal_topk.shape[:-1], self.n_freqs - topk,
                    device=x.device,
                    dtype=x.dtype,
                )
                seasonal_topk = torch.cat([seasonal_topk, pad], dim=-1)
        else:
            seasonal_topk = torch.zeros(x.size(0), self.in_channels, self.n_freqs, device=x.device, dtype=x.dtype)

        seasonal_flat = seasonal_topk.reshape(x.shape[0], -1)
        return torch.cat([slope, seasonal_flat], dim=-1)


class ARFeature(_BaseFeature):
    """Autoregressive coefficients fitted via least squares."""

    def __init__(self, in_channels: int, ar_order: int = 3, ar_regularizer: float = 1e-4) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ar_order = max(1, ar_order)
        self.ar_regularizer = ar_regularizer
        self.output_dim = in_channels * self.ar_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        coeffs = self._fit_ar_params(centered)
        return coeffs.reshape(x.shape[0], -1)

    def _fit_ar_params(self, series: torch.Tensor) -> torch.Tensor:
        B, L, C = series.shape
        order = min(self.ar_order, max(L - 1, 1))
        if L <= 1 or order <= 0:
            return torch.zeros(B, C, self.ar_order, device=series.device, dtype=series.dtype)

        target = series[:, order:, :].permute(0, 2, 1)
        lagged = []
        for lag in range(1, order + 1):
            lagged.append(series[:, order - lag:-lag, :])
        lag_tensor = torch.stack(lagged, dim=-1).permute(0, 3, 1, 2)
        lag_tensor = lag_tensor.permute(0, 3, 2, 1)

        Xt = lag_tensor.transpose(-1, -2)
        XtX = torch.matmul(Xt, lag_tensor)
        eye = torch.eye(order, device=series.device, dtype=series.dtype)
        XtX = XtX + self.ar_regularizer * eye.view(1, 1, order, order)
        Xty = torch.matmul(Xt, target.unsqueeze(-1))
        coeffs = torch.linalg.solve(XtX, Xty).squeeze(-1)

        if order < self.ar_order:
            pad = torch.zeros(B, C, self.ar_order - order, device=series.device, dtype=series.dtype)
            coeffs = torch.cat([coeffs, pad], dim=-1)
        return coeffs


FEATURE_BUILDERS = {
    'stats': StatsFeature,
    'trend': TrendSeasonFeature,
    'ar': ARFeature,
}


class SimpleTSFeature(nn.Module):
    """Composable feature extractor that mixes different statistical descriptors."""

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        d_feat: int,
        hidden: Optional[int] = None,
        method_names: Optional[List[str]] = None,
        n_freqs: int = 3,
        ar_order: int = 3,
        ar_regularizer: float = 1e-4,
    ) -> None:
        super().__init__()
        if method_names is None or len(method_names) == 0:
            method_names = ['stats', 'trend', 'ar']

        builder_kwargs: Dict[str, Dict] = {
            'stats': {},
            'trend': {'seq_len': seq_len, 'n_freqs': n_freqs},
            'ar': {'ar_order': ar_order, 'ar_regularizer': ar_regularizer},
        }

        self.blocks = nn.ModuleList()
        base_dim = 0
        for name in method_names:
            if name not in FEATURE_BUILDERS:
                raise ValueError(f"Unknown feature method: {name}")
            kwargs = builder_kwargs.get(name, {})
            block = FEATURE_BUILDERS[name](in_channels=in_channels, **kwargs)
            self.blocks.append(block)
            base_dim += block.output_dim

        if base_dim <= 0:
            raise ValueError("Feature extractor requires at least one method with positive output dimension.")

        if hidden is None:
            hidden = max(base_dim, d_feat)

        self.proj = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_feat),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [block(x) for block in self.blocks]
        stats = torch.cat(features, dim=-1)
        return self.proj(stats)
