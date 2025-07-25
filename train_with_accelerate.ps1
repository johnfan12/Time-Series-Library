# Windows PowerShell脚本用于使用accelerate进行多卡训练

# 设置加速器配置
$env:ACCELERATE_CONFIG_FILE = ".\accelerate_config.yaml"

# 使用accelerate launch启动训练
accelerate launch --config_file .\accelerate_config.yaml run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path .\dataset\ETT-small\ `
  --data_path ETTh1.csv `
  --model_id ETTh1_96_96 `
  --model TimesNet `
  --data ETTh1 `
  --features M `
  --seq_len 96 `
  --label_len 48 `
  --pred_len 96 `
  --e_layers 2 `
  --d_layers 1 `
  --factor 3 `
  --enc_in 7 `
  --dec_in 7 `
  --c_out 7 `
  --des 'Exp' `
  --d_model 64 `
  --d_ff 128 `
  --batch_size 32 `
  --learning_rate 0.0001 `
  --train_epochs 10 `
  --use_accelerate `
  --mixed_precision fp16
