# Accelerate多卡训练支持说明

本项目已添加了HuggingFace Accelerate多卡训练支持，可以轻松实现分布式训练。

## 安装依赖

```bash
pip install accelerate
```

## 配置Accelerate

### 方法1：使用accelerate config命令
```bash
accelerate config
```
按提示选择配置选项：
- Compute environment: LOCAL_MACHINE
- Distributed type: MULTI_GPU
- GPU数量：根据你的硬件设置
- Mixed precision: 根据需要选择(no/fp16/bf16)

### 方法2：使用提供的配置文件
项目中已包含`accelerate_config.yaml`配置文件，可以直接使用或根据需要修改：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'no'  # 可选: 'no', 'fp16', 'bf16'
num_machines: 1
num_processes: 2  # 使用的GPU数量
rdzv_backend: static
same_network: true
use_cpu: false
```

## 使用方法

### 1. 使用accelerate launch启动训练

**Linux/MacOS:**
```bash
accelerate launch --config_file ./accelerate_config.yaml run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --data ETTh1 \
  --use_accelerate \
  --mixed_precision fp16 \
  [其他参数...]
```

**Windows PowerShell:**
```powershell
accelerate launch --config_file ./accelerate_config.yaml run.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --model TimesNet `
  --data ETTh1 `
  --use_accelerate `
  --mixed_precision fp16 `
  [其他参数...]
```

### 2. 使用提供的脚本

**Linux/MacOS:**
```bash
chmod +x train_with_accelerate.sh
./train_with_accelerate.sh
```

**Windows:**
```powershell
./train_with_accelerate.ps1
```

## 新增参数说明

- `--use_accelerate`: 启用accelerate多卡训练支持
- `--mixed_precision`: 混合精度训练，可选值：no, fp16, bf16

## 支持的任务类型

所有实验类型都已支持accelerate：
- `long_term_forecast`: 长期预测
- `short_term_forecast`: 短期预测  
- `imputation`: 数据插值
- `classification`: 分类
- `anomaly_detection`: 异常检测

## 示例

### 长期预测任务
```bash
accelerate launch --config_file ./accelerate_config.yaml run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model TimesNet \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --use_accelerate \
  --mixed_precision fp16
```

### 分类任务
```bash
accelerate launch --config_file ./accelerate_config.yaml run.py \
  --task_name classification \
  --is_training 1 \
  --model TimesNet \
  --data UEA \
  --root_path ./dataset/UEA/ \
  --use_accelerate \
  --mixed_precision fp16
```

## 注意事项

1. 使用accelerate时，不需要设置`--use_multi_gpu`参数，accelerate会自动处理多GPU
2. 混合精度训练可以显著提升训练速度，建议在支持的硬件上使用
3. 确保所有GPU都有足够的显存
4. 可以通过修改`accelerate_config.yaml`中的`num_processes`来调整使用的GPU数量

## 性能对比

使用accelerate的优势：
- 更好的多GPU支持和负载均衡
- 内置混合精度训练支持
- 自动梯度同步和优化
- 更好的内存管理
- 支持更多的分布式训练策略

## 故障排除

1. 如果遇到CUDA内存不足，可以：
   - 减小batch_size
   - 启用gradient_checkpointing
   - 使用混合精度训练

2. 如果遇到进程同步问题：
   - 检查所有GPU是否可用
   - 确认网络配置正确
   - 重新生成accelerate配置
