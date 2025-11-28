"""Multi-dataset concatenated DataLoader for cross-dataset univariate training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data_provider.data_factory import data_dict


# Fixed time feature dimension for cross-dataset compatibility
# Using 4 features: month, day, weekday, hour (timeenc=0 style)
FIXED_TIME_FEAT_DIM = 4


@dataclass
class MultiDatasetSpec:
    """Specification for a single dataset in multi-dataset mode."""
    name: str
    data: str  # key for data_dict (ETTh1, custom, etc.)
    root_path: str
    data_path: str
    target: str = 'OT'
    freq: str = 'h'

    @classmethod
    def from_dict(cls, d: Dict) -> 'MultiDatasetSpec':
        return cls(**d)


def load_multi_dataset_specs(spec_path: Path) -> List[MultiDatasetSpec]:
    """Load dataset specifications from a JSON file."""
    with open(spec_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [MultiDatasetSpec.from_dict(item) for item in raw]
    raise ValueError("Multi-dataset spec must be a JSON array of dataset objects")


class DatasetWithUnifiedTimeFeatures(Dataset):
    """Wrapper that unifies time feature dimensions across datasets."""

    def __init__(self, base_dataset: Dataset, origin_idx: int, origin_name: str, 
                 target_time_dim: int = FIXED_TIME_FEAT_DIM):
        self.base_dataset = base_dataset
        self.origin_idx = origin_idx
        self.origin_name = origin_name
        self.target_time_dim = target_time_dim

    def __len__(self):
        return len(self.base_dataset)

    def _pad_or_truncate_time_features(self, time_feat: np.ndarray) -> np.ndarray:
        """Pad or truncate time features to target dimension."""
        if time_feat.ndim == 1:
            time_feat = time_feat.reshape(-1, 1)
        
        seq_len, feat_dim = time_feat.shape
        
        if feat_dim == self.target_time_dim:
            return time_feat
        elif feat_dim < self.target_time_dim:
            # Pad with zeros
            padding = np.zeros((seq_len, self.target_time_dim - feat_dim), dtype=time_feat.dtype)
            return np.concatenate([time_feat, padding], axis=1)
        else:
            # Truncate
            return time_feat[:, :self.target_time_dim]

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        # item is (seq_x, seq_y, seq_x_mark, seq_y_mark)
        seq_x, seq_y, seq_x_mark, seq_y_mark = item
        
        # Unify time feature dimensions
        seq_x_mark = self._pad_or_truncate_time_features(seq_x_mark)
        seq_y_mark = self._pad_or_truncate_time_features(seq_y_mark)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark


def create_multi_dataset(
    specs: List[MultiDatasetSpec],
    flag: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    args_template,
) -> Tuple[ConcatDataset, List[str]]:
    """Create a concatenated dataset from multiple dataset specs.

    Args:
        specs: List of dataset specifications
        flag: 'train', 'val', or 'test'
        seq_len: Input sequence length
        label_len: Label length for decoder
        pred_len: Prediction length
        args_template: Template namespace with common args (embed, etc.)

    Returns:
        Tuple of (ConcatDataset, list of dataset names)
    """
    datasets = []
    dataset_names = []

    timeenc = 0 if getattr(args_template, 'embed', 'timeF') != 'timeF' else 1

    for i, spec in enumerate(specs):
        Data = data_dict[spec.data]
        
        # Create a minimal args namespace for this dataset
        ds_args = type('Args', (), {
            'augmentation_ratio': 0,
        })()

        ds = Data(
            args=ds_args,
            root_path=spec.root_path,
            data_path=spec.data_path,
            flag=flag,
            size=[seq_len, label_len, pred_len],
            features='S',  # univariate for cross-dataset
            target=spec.target,
            timeenc=timeenc,
            freq=spec.freq,
            seasonal_patterns='monthly',
        )

        wrapped = DatasetWithUnifiedTimeFeatures(ds, i, spec.name)
        datasets.append(wrapped)
        dataset_names.append(spec.name)
        print(f"  {spec.name} [{flag}]: {len(ds)} samples")

    combined = ConcatDataset(datasets)
    return combined, dataset_names


def multi_dataset_provider(
    spec_path: Path,
    flag: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    args_template,
) -> Tuple[ConcatDataset, DataLoader, List[str]]:
    """High-level provider for multi-dataset training/evaluation.

    Args:
        spec_path: Path to JSON spec file
        flag: 'train', 'val', or 'test'
        seq_len, label_len, pred_len: Sequence lengths
        batch_size: Batch size
        num_workers: DataLoader workers
        args_template: Template namespace with common args

    Returns:
        Tuple of (dataset, dataloader, list of dataset names)
    """
    specs = load_multi_dataset_specs(spec_path)
    print(f"Loading {len(specs)} datasets for {flag}...")

    dataset, names = create_multi_dataset(
        specs, flag, seq_len, label_len, pred_len, args_template
    )

    shuffle = flag == 'train'
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"Total {flag} samples: {len(dataset)}")
    return dataset, loader, names
