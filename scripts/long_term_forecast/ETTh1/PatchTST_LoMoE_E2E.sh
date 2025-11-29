#!/bin/bash
# PatchTST_LoMoE_E2E on single dataset (ETTh1)
# End-to-End learned router without pre-computed clustering

export CUDA_VISIBLE_DEVICES=0

# Dataset
DATA=ETTh1
ROOT_PATH=./dataset/ETT-small/
DATA_PATH=ETTh1.csv

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
MOE_TOPK=2
ROUTER_TEMPERATURE=1.0
ROUTER_NOISE_STD=0.1
MOE_AUX_LOSS_COEFF=0.01

# Training hyperparameters
TRAIN_EPOCHS=30
LEARNING_RATE=0.0001
BATCH_SIZE=32
PATIENCE=5

echo "=== PatchTST_LoMoE_E2E on $DATA ==="
echo "num_experts=$NUM_EXPERTS, top_k=$MOE_TOPK, lora_rank=$LORA_RANK"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ${DATA}_lomoe_e2e_${SEQ_LEN}_${PRED_LEN} \
    --model PatchTST_LoMoE_E2E \
    --data $DATA \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --features M \
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
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_experts $NUM_EXPERTS \
    --lora_rank $LORA_RANK \
    --moe_topk $MOE_TOPK \
    --router_temperature $ROUTER_TEMPERATURE \
    --router_noise_std $ROUTER_NOISE_STD \
    --moe_aux_loss_coeff $MOE_AUX_LOSS_COEFF \
    --des 'LoMoE-E2E' \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --itr 1 \
    --use_gpu
