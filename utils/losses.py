# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

from typing import Optional

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


def _ensure_3d(tensor: t.Tensor) -> None:
    if tensor.dim() != 3:
        raise ValueError(
            f"Expected a tensor with shape [batch, length, channels], got {tensor.shape}"
        )


def _moving_average(sequence: t.Tensor, window_size: int) -> t.Tensor:
    if window_size <= 1:
        return sequence

    _ensure_3d(sequence)
    batch, length, channels = sequence.shape
    series = sequence.permute(0, 2, 1).reshape(batch * channels, 1, length)

    left_pad = (window_size - 1) // 2
    right_pad = window_size - 1 - left_pad
    kernel = t.ones(1, 1, window_size, device=sequence.device, dtype=sequence.dtype) / window_size

    padded = F.pad(series, (left_pad, right_pad), mode='replicate')
    trend = F.conv1d(padded, kernel)
    trend = trend.reshape(batch, channels, length).permute(0, 2, 1)
    return trend


def _trend_loss(y_true: t.Tensor, y_pred: t.Tensor, window: int) -> t.Tensor:
    if window <= 1:
        return t.zeros(1, device=y_true.device, dtype=y_true.dtype)
    true_trend = _moving_average(y_true, window)
    pred_trend = _moving_average(y_pred, window)
    return F.mse_loss(pred_trend, true_trend)


def _season_loss(y_true: t.Tensor, y_pred: t.Tensor, period: Optional[int]) -> t.Tensor:
    if period is None or period < 2:
        return t.zeros(1, device=y_true.device, dtype=y_true.dtype)

    _ensure_3d(y_true)
    batch, length, channels = y_true.shape
    trimmed = (length // period) * period
    if trimmed == 0:
        return t.zeros(1, device=y_true.device, dtype=y_true.dtype)

    true_cycles = y_true[:, :trimmed, :].reshape(batch, -1, period, channels)
    pred_cycles = y_pred[:, :trimmed, :].reshape(batch, -1, period, channels)

    true_pattern = true_cycles.mean(dim=1)
    pred_pattern = pred_cycles.mean(dim=1)
    return F.mse_loss(pred_pattern, true_pattern)


def _find_jump_indices(y_true: t.Tensor, horizon: int, threshold: float) -> t.Tensor:
    _ensure_3d(y_true)
    batch, length, _ = y_true.shape
    if horizon <= 0 or length < 2:
        return t.zeros((batch, length), device=y_true.device, dtype=t.bool)

    jump_mask = t.zeros((batch, length), device=y_true.device, dtype=t.bool)
    for step in range(1, horizon + 1):
        if step >= length:
            break
        delta = t.abs(y_true[:, step:, :] - y_true[:, :-step, :]).amax(dim=-1)
        pad = t.zeros((batch, step), device=y_true.device, dtype=delta.dtype)
        delta = t.cat([pad, delta], dim=1)
        jump_mask = jump_mask | (delta > threshold)
    return jump_mask


def _build_pre_jump_mask(jump_mask: t.Tensor, pre_days: int) -> t.Tensor:
    batch, length = jump_mask.shape
    if pre_days <= 0:
        return jump_mask.float()

    weight = jump_mask.float()
    max_offset = min(pre_days, max(0, length - 1))
    for offset in range(1, max_offset + 1):
        pad = t.zeros((batch, offset), device=jump_mask.device, dtype=jump_mask.dtype)
        shifted = t.cat([jump_mask[:, offset:], pad], dim=1)
        weight += shifted.float()
    return weight


def _jump_loss(
    y_true: t.Tensor,
    y_pred: t.Tensor,
    horizon: int,
    threshold: float,
    pre_days: int,
    base_weight: float,
    jump_scale: float,
) -> t.Tensor:
    jump_mask = _find_jump_indices(y_true, horizon, threshold)
    pre_mask = _build_pre_jump_mask(jump_mask, pre_days)
    weights = base_weight + jump_scale * pre_mask
    if t.all(weights <= 0):
        return t.zeros(1, device=y_true.device, dtype=y_true.dtype)
    weights = weights.clamp(min=1e-6).unsqueeze(-1)

    error = (y_pred - y_true) ** 2
    weighted_error = error * weights
    normalization = weights.sum().clamp(min=1.0)
    return weighted_error.sum() / normalization


class MultiObjectiveTimeSeriesLoss(nn.Module):
    """Weighted combination of point-wise, trend, seasonal, and jump-sensitive losses."""

    def __init__(
        self,
        *,
        period: Optional[int] = None,
        trend_window: int = 7,
        jump_horizon: int = 2,
        jump_threshold: float = 1.0,
        jump_pre_days: int = 5,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        delta: float = 1.0,
        jump_base_weight: float = 1.0,
        jump_scale: float = 5.0,
    ) -> None:
        super().__init__()
        self.period = period if period and period > 0 else None
        self.trend_window = max(1, trend_window)
        self.jump_horizon = max(0, jump_horizon)
        self.jump_threshold = jump_threshold
        self.jump_pre_days = max(0, jump_pre_days)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.jump_base_weight = jump_base_weight
        self.jump_scale = jump_scale

    def forward(self, forecast: t.Tensor, target: t.Tensor) -> t.Tensor:
        if forecast.shape != target.shape:
            raise ValueError(
                f"Predictions and targets must share the same shape, got {forecast.shape} vs {target.shape}"
            )
        _ensure_3d(target)

        mse = F.mse_loss(forecast, target)
        zero = t.zeros(1, device=target.device, dtype=target.dtype)
        trend = _trend_loss(target, forecast, self.trend_window) if self.beta != 0 else zero
        season = _season_loss(target, forecast, self.period) if self.gamma != 0 else zero
        jump = (
            _jump_loss(
                target,
                forecast,
                self.jump_horizon,
                self.jump_threshold,
                self.jump_pre_days,
                self.jump_base_weight,
                self.jump_scale,
            )
            if self.delta != 0
            else zero
        )

        total = self.alpha * mse + self.beta * trend + self.gamma * season + self.delta * jump
        return total
