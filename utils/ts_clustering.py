"""Clustering utilities for reduced time-series feature embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from sklearn.cluster import KMeans


ClusterMethod = Literal["kmeans"]


@dataclass
class ClusterConfig:
    method: ClusterMethod = "kmeans"
    n_clusters: int = 4
    random_state: Optional[int] = 0
    max_iter: int = 300
    n_init: int = 10


class TSClusterer:
    """Simple wrapper that exposes centroids and serialization helpers."""

    def __init__(self, cfg: Optional[ClusterConfig] = None):
        self.cfg = cfg or ClusterConfig()
        self._model = None
        self._fitted = False

    def _build_model(self):
        if self.cfg.method == "kmeans":
            self._model = KMeans(
                n_clusters=self.cfg.n_clusters,
                random_state=self.cfg.random_state,
                max_iter=self.cfg.max_iter,
                n_init=self.cfg.n_init,
            )
        else:
            raise ValueError(f"Unsupported cluster method: {self.cfg.method}")

    @property
    def model(self):
        if self._model is None:
            self._build_model()
        return self._model

    def fit(self, embeddings: np.ndarray) -> "TSClusterer":
        self.model.fit(embeddings)
        self._fitted = True
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Clusterer must be fitted before predict().")
        return self.model.predict(embeddings)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        labels = self.model.fit_predict(embeddings)
        self._fitted = True
        return labels

    @property
    def centroids(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Cannot access centroids before fitting.")
        if not hasattr(self.model, "cluster_centers_"):
            raise RuntimeError("Underlying model does not expose centroids.")
        return self.model.cluster_centers_

    def save(self, path: Path | str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save clusterer before fitting.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "state": self.model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "TSClusterer":
        payload = joblib.load(Path(path))
        instance = cls(payload["cfg"])
        instance._model = payload["state"]
        instance._fitted = True
        return instance
