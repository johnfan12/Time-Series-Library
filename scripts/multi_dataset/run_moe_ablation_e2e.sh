#!/bin/bash
# =============================================================================
# MoE Ablation Study: Compare Single Expert vs Multi-Expert (E2E Model)
# 
# This script runs two experiments:
# 1. Single Expert Baseline: num_experts=1, moe_topk=1 (equivalent to no MoE)
# 2. Multi-Expert MoE: num_experts=4, moe_topk=2 (full MoE)
#
# By comparing results, we can verify if MoE structure provides benefit.
# =============================================================================

export CUDA_VISIBLE_DEVICES=0

SPEC_PATH="scripts/cluster_prep/specs/multi_dataset_univariate.json"

# Common model hyperparameters
SEQ_LEN=96
PRED_LEN=96
D_MODEL=128
D_FF=256
N_HEADS=8
E_LAYERS=3
LORA_RANK=8

# Training hyperparameters
TRAIN_EPOCHS=30
LEARNING_RATE=0.0001
BATCH_SIZE=32
PATIENCE=5

# =============================================================================
# Experiment 1: Single Expert Baseline (No MoE effect)
# =============================================================================
echo ""
echo "=============================================================="
echo "  Experiment 1: Single Expert Baseline (num_experts=1)"
echo "  This is equivalent to PatchTST + single LoRA adapter"
echo "=============================================================="
echo ""

python -u run.py \
    --task_name multi_dataset_forecast \
    --is_training 1 \
    --model_id moe_ablation_single_expert_96_96 \
    --model PatchTST_LoMoE_E2E \
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
    --num_experts 1 \
    --lora_rank $LORA_RANK \
    --moe_topk 1 \
    --des 'MoE-Ablation-SingleExpert' \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --itr 1 \
    --use_gpu

echo ""
echo "Single Expert experiment completed."
echo ""

# =============================================================================
# Experiment 2: Multi-Expert MoE (Full MoE with learned routing)
# =============================================================================
echo ""
echo "=============================================================="
echo "  Experiment 2: Multi-Expert MoE (num_experts=4, top_k=2)"
echo "  Full MoE with learned neural network router"
echo "=============================================================="
echo ""

python -u run.py \
    --task_name multi_dataset_forecast \
    --is_training 1 \
    --model_id moe_ablation_multi_expert_96_96 \
    --model PatchTST_LoMoE_E2E \
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
    --num_experts 4 \
    --lora_rank $LORA_RANK \
    --moe_topk 2 \
    --router_temperature 1.0 \
    --router_noise_std 0.1 \
    --moe_aux_loss_coeff 0.01 \
    --des 'MoE-Ablation-MultiExpert' \
    --train_epochs $TRAIN_EPOCHS \
    --patience $PATIENCE \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --itr 1 \
    --use_gpu

echo ""
echo "Multi-Expert experiment completed."
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================================="
echo "  MoE Ablation Study Completed!"
echo "=============================================================="
echo ""
echo "Compare results in result_multi_dataset_forecast.txt:"
echo "  - moe_ablation_single_expert_96_96: Single LoRA (baseline)"
echo "  - moe_ablation_multi_expert_96_96: Full MoE (4 experts, top-2)"
echo ""
echo "If MoE is effective, multi-expert should show lower MSE/MAE."
echo ""
