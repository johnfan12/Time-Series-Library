import torch
import torch.nn as nn
from importlib import import_module
from typing import List, cast


class Model(nn.Module):
    """Tri-model ensemble that fuses PatchTST, iTransformer, and DLinear outputs."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.method = getattr(configs, 'ensemble_method', 'mean')
        trim_value = getattr(configs, 'ensemble_trim_ratio', 0.0)
        trim_float = 0.0 if trim_value is None else float(trim_value)
        self.trim_ratio = max(0.0, min(0.49, trim_float))

        default_models = ['PatchTST', 'iTransformer', 'DLinear']
        requested_models = getattr(configs, 'ensemble_models', None) or default_models
        if not isinstance(requested_models, (list, tuple)):
            raise ValueError('ensemble_models must be a sequence of model names')

        self.base_model_names: List[str] = list(requested_models)
        if len(self.base_model_names) == 0:
            raise ValueError('At least one base model is required for the ensemble')

        self.models = nn.ModuleList([self._build_single_model(name) for name in self.base_model_names])

        weights = getattr(configs, 'ensemble_weights', None)
        if weights is not None:
            if len(weights) != len(self.base_model_names):
                raise ValueError('ensemble_weights length must match ensemble_models length')
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
        else:
            weight_tensor = torch.ones(len(self.base_model_names), dtype=torch.float32)
        weight_tensor = weight_tensor / weight_tensor.sum()
        self.register_buffer('weight_buffer', weight_tensor)

    def _build_single_model(self, name: str) -> nn.Module:
        try:
            module = import_module(f'models.{name}')
        except ModuleNotFoundError as exc:
            raise ValueError(f'Failed to import base model "{name}"') from exc
        if not hasattr(module, 'Model'):
            raise ValueError(f'Model "{name}" does not expose a Model class')
        return module.Model(self.configs)

    def _aggregate(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(outputs, dim=0)
        method = self.method.lower()
        if method == 'mean':
            return stacked.mean(dim=0)
        if method == 'median':
            return stacked.median(dim=0).values
        if method == 'trimmed_mean':
            trim = int(self.trim_ratio * stacked.size(0))
            trim = min(trim, (stacked.size(0) - 1) // 2)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim: stacked.size(0) - trim] if trim > 0 else sorted_vals
            return trimmed.mean(dim=0)
        if method == 'weighted':
            weights = cast(torch.Tensor, self.weight_buffer).to(stacked.device)
            if stacked.ndim == 4:
                weights = torch.reshape(weights, (-1, 1, 1, 1))
            elif stacked.ndim == 3:
                weights = torch.reshape(weights, (-1, 1, 1))
            elif stacked.ndim == 2:
                weights = torch.reshape(weights, (-1, 1))
            else:
                raise ValueError('Unsupported output dimensions for weighted ensemble reduction')
            weighted = stacked * weights
            return weighted.sum(dim=0)
        raise ValueError(f'Unsupported ensemble_method "{self.method}"')

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        outputs = []
        for model in self.models:
            outputs.append(model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask))
        return self._aggregate(outputs)
