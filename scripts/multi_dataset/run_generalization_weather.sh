#!/bin/bash
# Generalization experiment: Train without Weather, test on Weather
# This tests the model's ability to generalize to unseen datasets

export CUDA_VISIBLE_DEVICES=0

# Use the generalization spec (Weather is test_only)
SPEC_PATH="scripts/cluster_prep/specs/multi_dataset_generalization_weather.json"

# Model hyperparameters
SEQ_LEN=96
PRED_LEN=96
D_MODEL=128
D_FF=256
N_HEADS=8
E_LAYERS=3

# Training hyperparameters
TRAIN_EPOCHS=30
LEARNING_RATE=0.0001
BATCH_SIZE=32
PATIENCE=5

echo "=== Generalization Experiment: Weather as unseen test set ==="
echo "Training on: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Traffic, Exchange"
echo "Testing on: ALL datasets (including unseen Weather)"

python -u run.py \
    --task_name multi_dataset_forecast \
    --is_training 1 \
    --model_id generalization_weather_patchtst_96_96 \
    --model PatchTST \
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
    --des 'Generalization-Weather-Unseen' \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --itr 1 \
    --use_gpu
