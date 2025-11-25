"""Pipeline to create cluster artifacts for stats-driven LoMoE routing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from data_provider.data_factory import data_provider
from utils.ts_stats import FeatureExtractionConfig, batch_extract_ts_features
from utils.ts_dim_reduction import ReducerConfig, TSFeatureReducer
from utils.ts_clustering import ClusterConfig, TSClusterer


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


def _collect_series(loader) -> np.ndarray:
    sequences: List[np.ndarray] = []
    for batch_x, _, _, _ in tqdm(loader, desc="Collecting train batches"):
        sequences.append(batch_x.detach().cpu().numpy())
    return np.concatenate(sequences, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cluster artifacts for stats router")
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/ETTh1/')
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

    parser.add_argument('--feature_max_acf', type=int, default=6)
    parser.add_argument('--feature_topk_fft', type=int, default=3)
    parser.add_argument('--feature_poly_order', type=int, default=1)
    parser.add_argument('--feature_clip', type=float, default=5.0)

    parser.add_argument('--reducer_components', type=int, default=16)
    parser.add_argument('--cluster_num', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./cluster_artifacts/ETTh1')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
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

    reducer = TSFeatureReducer(ReducerConfig(n_components=args.reducer_components))
    reduced = reducer.fit_transform(all_features)
    print(f"Reduced embeddings shape: {reduced.shape}")

    clusterer = TSClusterer(ClusterConfig(n_clusters=args.cluster_num))
    clusterer.fit(reduced)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    reducer.save(save_dir / 'reducer.joblib')
    clusterer.save(save_dir / 'cluster.joblib')

    with open(save_dir / 'feature_cfg.json', 'w', encoding='utf-8') as f:
        json.dump(asdict(feat_cfg), f, indent=2)

    print(f"Artifacts saved to {save_dir}")


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
