"""Dimensionality reduction helpers for time-series statistical embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA


ReducerMethod = Literal["pca"]


@dataclass
class ReducerConfig:
    method: ReducerMethod = "pca"
    n_components: int = 16
    whiten: bool = False
    random_state: Optional[int] = 0


class TSFeatureReducer:
    """Wrapper around common reducers with save/load utilities."""

    def __init__(self, cfg: Optional[ReducerConfig] = None):
        self.cfg = cfg or ReducerConfig()
        self._model = None
        self._fitted = False

    def _build_model(self):
        if self.cfg.method == "pca":
            self._model = PCA(
                n_components=self.cfg.n_components,
                whiten=self.cfg.whiten,
                random_state=self.cfg.random_state,
            )
        else:
            raise ValueError(f"Unsupported reducer method: {self.cfg.method}")

    @property
    def model(self):
        if self._model is None:
            self._build_model()
        return self._model

    def fit(self, features: np.ndarray) -> "TSFeatureReducer":
        self.model.fit(features)
        self._fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Reducer must be fitted before calling transform().")
        return self.model.transform(features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        transformed = self.model.fit_transform(features)
        self._fitted = True
        return transformed

    def save(self, path: Path | str) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save reducer before fitting.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "state": self.model,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> "TSFeatureReducer":
        payload = joblib.load(Path(path))
        instance = cls(payload["cfg"])
        instance._model = payload["state"]
        instance._fitted = True
        return instance
