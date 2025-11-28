#!/bin/bash
# Multi-dataset univariate long-term forecasting with PatchTST_LoMoE
# This script trains on all datasets jointly and tests on each separately

export CUDA_VISIBLE_DEVICES=0

# 1) First run clustering preparation to generate cluster artifacts
# Uncomment and run this first if cluster artifacts don't exist
# python scripts/cluster_prep/prepare_clusters.py \
#     --multi_dataset_spec scripts/cluster_prep/specs/multi_dataset_univariate.json \
#     --n_clusters 4 \
#     --seq_len 96 \
#     --output_dir ./cluster_artifacts/multi_dataset_univariate

SPEC_PATH="scripts/cluster_prep/specs/multi_dataset_univariate.json"
CLUSTER_DIR="./cluster_artifacts/multi_dataset_univariate"

# Model hyperparameters
SEQ_LEN=96
PRED_LEN=96
D_MODEL=128
D_FF=256
N_HEADS=8
E_LAYERS=3

# LoMoE hyperparameters
NUM_EXPERTS=4
LORA_RANK=8

# Training hyperparameters
TRAIN_EPOCHS=30
LEARNING_RATE=0.0001
BATCH_SIZE=32
PATIENCE=5

# Two-phase training: warmup with single expert, then finetune with cluster routing
# Set LOMOE_WARMUP_EPOCHS=0 to disable warmup phase (always use routing)
LOMOE_WARMUP_EPOCHS=10
PHASE2_LR_SCALE=0.1

python -u run.py \
    --task_name multi_dataset_forecast \
    --is_training 1 \
    --model_id multi_dataset_univariate_96_96 \
    --model PatchTST_LoMoE \
    --data custom \
    --multi_dataset_spec $SPEC_PATH \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --features S \
    --target OT \
    --seq_len $SEQ_LEN \
    --label_len 48 \
    --pred_len $PRED_LEN \
    --e_layers $E_LAYERS \
    --d_layers 1 \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --n_heads $N_HEADS \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --num_experts $NUM_EXPERTS \
    --lora_rank $LORA_RANK \
    --cluster_artifact_dir $CLUSTER_DIR \
    --lomoe_warmup_epochs $LOMOE_WARMUP_EPOCHS \
    --lomoe_freeze_backbone_after_warmup \
    --lomoe_phase2_lr_scale $PHASE2_LR_SCALE \
    --des 'MultiDataset-Univariate' \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --itr 1 \
    --use_gpu
