#!/bin/bash

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_LoMoE \
  --model PatchTST_LoMoE \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp_LoMoE_PoC \
  --d_model 128 \
  --d_ff 256 \
  --num_experts 4 \
  --lora_rank 8 \
  --moe_topk 2 \
  --moe_aux_loss_coeff 0.01 \
  --itr 1
