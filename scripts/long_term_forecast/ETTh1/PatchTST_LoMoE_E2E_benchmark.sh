#!/bin/bash
# PatchTST_LoMoE_E2E on ETTh1 - Full benchmark with multiple prediction lengths
# Compares: PatchTST baseline vs LoMoE_E2E (single expert) vs LoMoE_E2E (multi expert)

export CUDA_VISIBLE_DEVICES=0

# Dataset
DATA=ETTh1
ROOT_PATH=./dataset/ETT-small/
DATA_PATH=ETTh1.csv

# Common hyperparameters
SEQ_LEN=96
D_MODEL=128
D_FF=256
N_HEADS=8
E_LAYERS=3
LORA_RANK=8
TRAIN_EPOCHS=30
LEARNING_RATE=0.0001
BATCH_SIZE=32
PATIENCE=5

for PRED_LEN in 96 192 336 720; do
    echo ""
    echo "=============================================================="
    echo "  Prediction Length: $PRED_LEN"
    echo "=============================================================="
    
    # -----------------------------------------------------------------
    # 1. PatchTST Baseline
    # -----------------------------------------------------------------
    echo ""
    echo ">>> [1/3] PatchTST Baseline"
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id ${DATA}_patchtst_${SEQ_LEN}_${PRED_LEN} \
        --model PatchTST \
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
        --des 'PatchTST-Baseline' \
        --train_epochs $TRAIN_EPOCHS \
        --patience $PATIENCE \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --itr 1 \
        --use_gpu

    # -----------------------------------------------------------------
    # 2. LoMoE_E2E Single Expert (ablation baseline)
    # -----------------------------------------------------------------
    echo ""
    echo ">>> [2/3] LoMoE_E2E Single Expert (num_experts=1)"
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id ${DATA}_lomoe_e2e_1expert_${SEQ_LEN}_${PRED_LEN} \
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
        --num_experts 1 \
        --lora_rank $LORA_RANK \
        --moe_topk 1 \
        --des 'LoMoE-E2E-1Expert' \
        --train_epochs $TRAIN_EPOCHS \
        --patience $PATIENCE \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --itr 1 \
        --use_gpu

    # -----------------------------------------------------------------
    # 3. LoMoE_E2E Multi Expert (full MoE)
    # -----------------------------------------------------------------
    echo ""
    echo ">>> [3/3] LoMoE_E2E Multi Expert (num_experts=4, top_k=2)"
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id ${DATA}_lomoe_e2e_4expert_${SEQ_LEN}_${PRED_LEN} \
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
        --num_experts 4 \
        --lora_rank $LORA_RANK \
        --moe_topk 2 \
        --router_temperature 1.0 \
        --router_noise_std 0.1 \
        --moe_aux_loss_coeff 0.01 \
        --des 'LoMoE-E2E-4Expert' \
        --train_epochs $TRAIN_EPOCHS \
        --patience $PATIENCE \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --itr 1 \
        --use_gpu

done

echo ""
echo "=============================================================="
echo "  Benchmark completed! Check result_long_term_forecast.txt"
echo "=============================================================="
