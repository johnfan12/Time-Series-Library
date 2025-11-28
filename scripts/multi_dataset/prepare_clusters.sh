#!/bin/bash
# Prepare cluster artifacts for multi-dataset univariate training
# This script should be run BEFORE training to generate cluster centroids

cd "$(dirname "$0")/../.."

SPEC_PATH="scripts/cluster_prep/specs/multi_dataset_univariate.json"
OUTPUT_DIR="./cluster_artifacts/multi_dataset_univariate"
N_CLUSTERS=4
SEQ_LEN=96

echo "Preparing cluster artifacts for multi-dataset univariate training..."
echo "  Spec path: $SPEC_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  N clusters: $N_CLUSTERS"
echo "  Seq len: $SEQ_LEN"

python scripts/cluster_prep/prepare_clusters.py \
    --multi_dataset_spec "$SPEC_PATH" \
    --n_clusters $N_CLUSTERS \
    --seq_len $SEQ_LEN \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "Cluster artifacts saved to $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
else
    echo "Failed to prepare cluster artifacts"
    exit 1
fi
