"""Utility helpers for extracting lightweight statistical descriptors from time series.

These descriptors are later used for clustering-driven LoMoE routing. The focus is on
producing stable, low-dimensional embeddings that summarize trend, dispersion and
frequency cues without assuming any specific dataset layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


@dataclass
class FeatureExtractionConfig:
    """Configuration flags for statistical feature extraction."""

    max_acf_lag: int = 6
    top_k_fft: int = 3
    poly_order: int = 1
    time_axis: int = 0
    clip_value: Optional[float] = None
    eps: float = 1e-6


def _prepare_series(series: np.ndarray, cfg: FeatureExtractionConfig) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 2:
        if cfg.time_axis == 0:
            arr = arr.T
        elif cfg.time_axis != 1:
            raise ValueError("time_axis must be 0 or 1")
    else:
        raise ValueError("Expected series with 1 or 2 dims: (seq_len,) or (seq_len, n_vars)")

    if cfg.clip_value is not None:
        np.clip(arr, -cfg.clip_value, cfg.clip_value, out=arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=cfg.clip_value or 0.0, neginf=-(cfg.clip_value or 0.0))
    return arr


def _autocorr_features(x: np.ndarray, max_lag: int, eps: float) -> np.ndarray:
    x = x - x.mean()
    denom = np.dot(x, x) + eps
    feats: List[float] = []
    for lag in range(1, max_lag + 1):
        if lag >= x.shape[-1]:
            feats.append(0.0)
            continue
        num = np.dot(x[:-lag], x[lag:])
        feats.append(float(num / denom))
    return np.asarray(feats, dtype=np.float32)


def _fft_energy_features(x: np.ndarray, top_k: int, eps: float) -> Tuple[float, float]:
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    total = float(power.sum() + eps)
    if power.size <= 1:
        return 0.0, 0.0
    sorted_power = np.sort(power[1:])  # ignore DC
    dominant = float(sorted_power[-1]) if sorted_power.size else 0.0
    top_band = float(sorted_power[-top_k:].sum()) if sorted_power.size else 0.0
    return dominant / total, top_band / total


def _trend_slope(x: np.ndarray, order: int) -> float:
    idx = np.arange(x.shape[-1], dtype=np.float32)
    degree = max(1, order)
    coeffs = np.polyfit(idx, x, deg=degree)
    return float(coeffs[-2]) if degree >= 1 else 0.0


def extract_ts_features(series: np.ndarray, cfg: Optional[FeatureExtractionConfig] = None) -> np.ndarray:
    """Return a compact feature vector summarizing a single multivariate series."""

    cfg = cfg or FeatureExtractionConfig()
    arr = _prepare_series(series, cfg)
    per_var_features: List[np.ndarray] = []
    for var_series in arr:
        mean = float(var_series.mean())
        std = float(var_series.std() + cfg.eps)
        skew = float(sp_stats.skew(var_series)) if var_series.size > 2 else 0.0
        kurt = float(sp_stats.kurtosis(var_series)) if var_series.size > 3 else 0.0
        acf = _autocorr_features(var_series, cfg.max_acf_lag, cfg.eps)
        dom_ratio, topk_ratio = _fft_energy_features(var_series, cfg.top_k_fft, cfg.eps)
        slope = _trend_slope(var_series, cfg.poly_order)
        feature_vec = np.concatenate([
            np.asarray(
                [
                    mean,
                    std,
                    float(var_series.min()),
                    float(var_series.max()),
                    float(np.median(var_series)),
                    skew,
                    kurt,
                    slope,
                    dom_ratio,
                    topk_ratio,
                ],
                dtype=np.float32,
            ),
            acf,
        ])
        per_var_features.append(feature_vec)

    stacked = np.stack(per_var_features, axis=0)
    summary = np.concatenate([
        stacked.mean(axis=0),
        stacked.std(axis=0),
    ])
    return summary.astype(np.float32)


def batch_extract_ts_features(batch: Sequence[np.ndarray], cfg: Optional[FeatureExtractionConfig] = None) -> np.ndarray:
    """Vectorize feature extraction across a batch of sequences."""

    features = [extract_ts_features(sample, cfg) for sample in batch]
    return np.stack(features, axis=0)


def feature_names(cfg: Optional[FeatureExtractionConfig] = None) -> List[str]:
    cfg = cfg or FeatureExtractionConfig()
    base_names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "skew",
        "kurtosis",
        "trend_slope",
        "fft_dom_ratio",
        "fft_topk_ratio",
    ]
    acf_names = [f"acf_lag_{lag}" for lag in range(1, cfg.max_acf_lag + 1)]
    single_names = base_names + acf_names
    return [f"avg_{name}" for name in single_names] + [f"std_{name}" for name in single_names]
