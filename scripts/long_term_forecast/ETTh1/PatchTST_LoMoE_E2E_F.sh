#!/bin/bash

# PatchTST_LoMoE_E2E_F on ETTh1 - using time series statistical features for routing
# This version uses statistical features (mean, std, trend, ACF, FFT) instead of encoder output

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST_LoMoE_E2E_F

# Common settings
seq_len=336
patch_len=16
stride=8
d_model=128
d_ff=256
n_heads=4
e_layers=3
dropout=0.3
batch_size=128
learning_rate=0.0001
train_epochs=100
patience=20

# LoMoE settings (similar to E2E model)
num_experts=4
lora_rank=8
lora_alpha=16
moe_topk=2
moe_aux_loss_coeff=0.01
router_temperature=1.0
router_noise_std=0.1

# E2E_F specific: statistical feature extraction settings
router_feat_acf_lags=5
router_feat_fft_topk=3

# pred_len variants
for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --patch_len $patch_len \
  --stride $stride \
  --dropout $dropout \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --num_experts $num_experts \
  --lora_rank $lora_rank \
  --lora_alpha $lora_alpha \
  --moe_topk $moe_topk \
  --moe_aux_loss_coeff $moe_aux_loss_coeff \
  --router_temperature $router_temperature \
  --router_noise_std $router_noise_std \
  --router_feat_acf_lags $router_feat_acf_lags \
  --router_feat_fft_topk $router_feat_fft_topk \
  --router_feat_include_trend \
  --itr 1

done
