"""Pipeline to create cluster artifacts for stats-driven LoMoE routing.

Supports:
  1. Single-dataset clustering (original behavior)
  2. Multi-dataset univariate clustering via --multi_dataset_spec JSON file
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data_provider.data_factory import data_provider
from utils.ts_stats import FeatureExtractionConfig, batch_extract_ts_features
from utils.ts_clustering import ClusterConfig, TSClusterer


# ---------------------------------------------------------------------------
# Dataset specification helpers
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    """Configuration for a single dataset in multi-dataset mode."""
    name: str
    data: str  # key for data_dict (ETTh1, custom, etc.)
    root_path: str
    data_path: str
    target: str = 'OT'
    freq: str = 'h'

    @classmethod
    def from_dict(cls, d: Dict) -> 'DatasetSpec':
        return cls(**d)


def load_multi_dataset_specs(spec_path: Path) -> List[DatasetSpec]:
    """Load dataset specifications from a JSON file."""
    with open(spec_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [DatasetSpec.from_dict(item) for item in raw]
    raise ValueError("Multi-dataset spec must be a JSON array of dataset objects")


# ---------------------------------------------------------------------------
# Single-dataset helpers (original)
# ---------------------------------------------------------------------------
def _get_dataset_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        data=cli_args.data,
        root_path=cli_args.root_path,
        data_path=cli_args.data_path,
        features=cli_args.features,
        target=cli_args.target,
        seq_len=cli_args.seq_len,
        label_len=cli_args.label_len,
        pred_len=cli_args.pred_len,
        seasonal_patterns=cli_args.seasonal_patterns,
        task_name=cli_args.task_name,
        embed=cli_args.embed,
        freq=cli_args.freq,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        augmentation_ratio=cli_args.augmentation_ratio,
    )


def _get_dataset_args_from_spec(
    spec: DatasetSpec,
    seq_len: int,
    label_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
) -> argparse.Namespace:
    """Build dataset args from a DatasetSpec (univariate mode)."""
    return argparse.Namespace(
        data=spec.data,
        root_path=spec.root_path,
        data_path=spec.data_path,
        features='S',  # univariate for cross-dataset
        target=spec.target,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        seasonal_patterns='monthly',
        task_name='long_term_forecast',
        embed='timeF',
        freq=spec.freq,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_ratio=0.0,
    )


def _collect_series(loader, desc: str = "Collecting batches") -> np.ndarray:
    sequences: List[np.ndarray] = []
    for batch_x, _, _, _ in tqdm(loader, desc=desc):
        sequences.append(batch_x.detach().cpu().numpy())
    return np.concatenate(sequences, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cluster artifacts for stats router (single or multi-dataset)"
    )
    # Single-dataset mode params
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='monthly')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--augmentation_ratio', type=float, default=0.0)

    # Feature extraction params
    parser.add_argument('--feature_max_acf', type=int, default=6)
    parser.add_argument('--feature_topk_fft', type=int, default=3)
    parser.add_argument('--feature_poly_order', type=int, default=1)
    parser.add_argument('--feature_clip', type=float, default=5.0)

    # Clustering params
    parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters')
    parser.add_argument('--cluster_num', type=int, default=None, help='[deprecated] alias for --n_clusters')
    parser.add_argument('--output_dir', type=str, default='./cluster_artifacts/ETTh1', help='output directory')
    parser.add_argument('--save_dir', type=str, default=None, help='[deprecated] alias for --output_dir')

    # Multi-dataset mode
    parser.add_argument(
        '--multi_dataset_spec',
        type=str,
        default=None,
        help='Path to JSON file specifying multiple datasets for joint univariate clustering'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main routines
# ---------------------------------------------------------------------------
def run_single_dataset(args: argparse.Namespace) -> None:
    """Original single-dataset clustering."""
    dataset_args = _get_dataset_args(args)
    dataset, loader = data_provider(dataset_args, flag='train')
    print(f"Loaded dataset with {len(dataset)} samples")

    series = _collect_series(loader)
    print(f"Collected series array shape: {series.shape}")

    feat_cfg = FeatureExtractionConfig(
        max_acf_lag=args.feature_max_acf,
        top_k_fft=args.feature_topk_fft,
        poly_order=args.feature_poly_order,
        clip_value=args.feature_clip,
    )

    all_features = batch_extract_ts_features(list(series), feat_cfg)
    print(f"Extracted features of shape: {all_features.shape}")

    clusterer = TSClusterer(ClusterConfig(n_clusters=args.n_clusters))
    clusterer.fit(all_features)

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    clusterer.save(save_dir / 'cluster.joblib')

    with open(save_dir / 'feature_cfg.json', 'w', encoding='utf-8') as f:
        json.dump(asdict(feat_cfg), f, indent=2)

    print(f"Artifacts saved to {save_dir}")


def run_multi_dataset(args: argparse.Namespace) -> None:
    """Multi-dataset univariate clustering.
    
    Collects univariate (--features S) training data from each dataset
    specified in the JSON spec, concatenates series, extracts features,
    and fits a single clusterer across all datasets.
    """
    spec_path = Path(args.multi_dataset_spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"Multi-dataset spec not found: {spec_path}")

    specs = load_multi_dataset_specs(spec_path)
    print(f"Loaded {len(specs)} dataset specs from {spec_path}")

    all_series: List[np.ndarray] = []
    dataset_labels: List[str] = []

    for spec in specs:
        print(f"\n--- Loading {spec.name} ---")
        ds_args = _get_dataset_args_from_spec(
            spec,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        dataset, loader = data_provider(ds_args, flag='train')
        print(f"  {spec.name}: {len(dataset)} samples")

        series = _collect_series(loader, desc=f"  Collecting {spec.name}")
        # series shape: [N, seq_len, 1] for univariate
        all_series.append(series)
        dataset_labels.extend([spec.name] * series.shape[0])

    combined = np.concatenate(all_series, axis=0)
    print(f"\nCombined series shape: {combined.shape}")

    feat_cfg = FeatureExtractionConfig(
        max_acf_lag=args.feature_max_acf,
        top_k_fft=args.feature_topk_fft,
        poly_order=args.feature_poly_order,
        clip_value=args.feature_clip,
    )

    all_features = batch_extract_ts_features(list(combined), feat_cfg)
    print(f"Extracted features of shape: {all_features.shape}")

    clusterer = TSClusterer(ClusterConfig(n_clusters=args.n_clusters))
    clusterer.fit(all_features)

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    clusterer.save(save_dir / 'cluster.joblib')

    with open(save_dir / 'feature_cfg.json', 'w', encoding='utf-8') as f:
        json.dump(asdict(feat_cfg), f, indent=2)

    # Save dataset list for reference
    with open(save_dir / 'datasets.json', 'w', encoding='utf-8') as f:
        json.dump([asdict(s) for s in specs], f, indent=2)

    # Optionally save per-sample dataset labels for analysis
    np.save(save_dir / 'dataset_labels.npy', np.array(dataset_labels))

    print(f"\nArtifacts saved to {save_dir}")
    print(f"  - cluster.joblib")
    print(f"  - feature_cfg.json")
    print(f"  - datasets.json (list of datasets)")
    print(f"  - dataset_labels.npy (per-sample dataset origin)")


def main():
    args = parse_args()
    # Handle deprecated alias arguments
    if args.cluster_num is not None and args.n_clusters == 4:
        args.n_clusters = args.cluster_num
        print(f"[Warning] --cluster_num is deprecated, use --n_clusters instead")
    if args.save_dir is not None and args.output_dir == './cluster_artifacts/ETTh1':
        args.output_dir = args.save_dir
        print(f"[Warning] --save_dir is deprecated, use --output_dir instead")

    if args.multi_dataset_spec:
        run_multi_dataset(args)
    else:
        run_single_dataset(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
