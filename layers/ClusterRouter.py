"""Routing modules that leverage precomputed statistical clusters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from utils.ts_dim_reduction import TSFeatureReducer
from utils.ts_stats import FeatureExtractionConfig, batch_extract_ts_features
from utils.ts_clustering import TSClusterer


@dataclass
class ClusterRouterArtifacts:
    reducer_path: Path
    cluster_path: Path
    feature_cfg: FeatureExtractionConfig


class ClusterDistanceRouter(nn.Module):
    """Computes routing weights via distances to stored centroids."""

    def __init__(
        self,
        centroids: torch.Tensor,
        top_k: int = 2,
        temperature: float = 1.0,
        metric: str = "euclidean",
    ) -> None:
        super().__init__()
        if centroids.ndim != 2:
            raise ValueError("centroids must be of shape [num_clusters, embed_dim]")
        self.register_buffer("centroids", centroids.float())
        self.num_experts = centroids.shape[0]
        self.embed_dim = centroids.shape[1]
        self.top_k = max(1, min(top_k, self.num_experts))
        self.temperature = max(1e-6, temperature)
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        self.metric = metric

    def _compute_distance(self, x: torch.Tensor) -> torch.Tensor:
        if self.metric == "euclidean":
            # torch.cdist expects same dimensionality
            return torch.cdist(x, self.centroids, p=2)
        x_norm = nn.functional.normalize(x, dim=-1)
        cent_norm = nn.functional.normalize(self.centroids, dim=-1)
        cosine_sim = x_norm @ cent_norm.t()
        return 1.0 - cosine_sim  # cosine distance

    def forward(self, embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeds.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embed_dim}, got {embeds.shape[-1]}"
            )
        dists = self._compute_distance(embeds)
        scaled = -dists / self.temperature
        probs = nn.functional.softmax(scaled, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        denom = torch.clamp(topk_weights.sum(dim=-1, keepdim=True), min=1e-6)
        topk_weights = topk_weights / denom
        return topk_weights, topk_indices, probs


class StatsClusterRouter:
    """High-level helper that translates raw series into routing weights."""

    def __init__(
        self,
        artifacts: ClusterRouterArtifacts,
        top_k: int = 2,
        temperature: float = 1.0,
        metric: str = "euclidean",
        device: Optional[torch.device] = None,
    ) -> None:
        reducer = TSFeatureReducer.load(artifacts.reducer_path)
        clusterer = TSClusterer.load(artifacts.cluster_path)
        centroids = torch.from_numpy(clusterer.centroids)
        self.cluster_router = ClusterDistanceRouter(
            centroids=centroids,
            top_k=top_k,
            temperature=temperature,
            metric=metric,
        )
        self.reducer = reducer
        self.feature_cfg = artifacts.feature_cfg
        self.device = device or torch.device("cpu")
        self.cluster_router.to(self.device)

    def _series_to_embeddings(self, series: torch.Tensor) -> torch.Tensor:
        batch_np = series.detach().cpu().numpy()
        feats = batch_extract_ts_features(list(batch_np), self.feature_cfg)
        reduced = self.reducer.transform(feats)
        return torch.from_numpy(reduced).to(self.device, dtype=torch.float32)

    def route(self, series: torch.Tensor):
        embeds = self._series_to_embeddings(series)
        return self.cluster_router(embeds)
