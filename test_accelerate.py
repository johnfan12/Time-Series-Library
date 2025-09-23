#!/usr/bin/env python3
"""
测试accelerate集成的简单脚本
"""
import argparse
import os
import sys

def test_accelerate_import():
    """测试accelerate是否正确安装"""
    try:
        from accelerate import Accelerator
        print("✓ Accelerate导入成功")
        return True
    except ImportError as e:
        print(f"✗ Accelerate导入失败: {e}")
        print("请运行: pip install accelerate")
        return False

def test_torch_import():
    """测试PyTorch是否正确安装"""
    try:
        import torch
        print(f"✓ PyTorch导入成功，版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("! CUDA不可用")
        return True
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False

def test_model_import():
    """测试模型导入"""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        print("✓ 实验类导入成功")
        return True
    except ImportError as e:
        print(f"✗ 实验类导入失败: {e}")
        return False

def create_test_args():
    """创建测试用的参数"""
    class TestArgs:
        def __init__(self):
            # 基本配置
            self.task_name = 'long_term_forecast'
            self.is_training = 1
            self.model_id = 'test'
            self.model = 'TimesNet'
            
            # 数据配置
            self.data = 'ETTh1'
            self.root_path = './dataset/ETT-small/'
            self.data_path = 'ETTh1.csv'
            self.features = 'M'
            self.target = 'OT'
            self.freq = 'h'
            self.checkpoints = './checkpoints/'
            
            # 模型配置
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
            self.enc_in = 7
            self.dec_in = 7
            self.c_out = 7
            self.d_model = 64
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 128
            self.factor = 3
            self.embed = 'timeF'
            self.dropout = 0.1
            self.activation = 'gelu'
            
            # 训练配置
            self.train_epochs = 1
            self.batch_size = 8
            self.learning_rate = 0.0001
            self.patience = 3
            self.des = 'test'
            self.loss = 'MSE'
            self.lradj = 'type1'
            
            # GPU配置
            self.use_gpu = True
            self.gpu = 0
            self.gpu_type = 'cuda'
            self.use_multi_gpu = False
            self.devices = '0,1'
            
            # Accelerate配置
            self.use_accelerate = True
            self.mixed_precision = 'no'
            
            # 其他配置
            self.use_amp = False
            self.num_workers = 0
            self.itr = 1
            self.top_k = 5
            self.num_kernels = 6
            self.moving_avg = 25
            self.distil = True
            self.channel_independence = 1
            self.decomp_method = 'moving_avg'
            self.use_norm = 1
            self.down_sampling_layers = 0
            self.down_sampling_window = 1
            self.down_sampling_method = None
            self.seg_len = 96
            self.expand = 2
            self.d_conv = 4
            
    return TestArgs()

def test_accelerate_initialization():
    """测试accelerate初始化"""
    try:
        from accelerate import Accelerator
        
        # 测试不同的混合精度设置
        for mixed_precision in ['no', 'fp16']:
            try:
                accelerator = Accelerator(mixed_precision=mixed_precision)
                print(f"✓ Accelerator初始化成功 (mixed_precision={mixed_precision})")
                print(f"  设备: {accelerator.device}")
                print(f"  进程数: {accelerator.num_processes}")
                print(f"  是否主进程: {accelerator.is_main_process}")
                break
            except Exception as e:
                print(f"! Accelerator初始化失败 (mixed_precision={mixed_precision}): {e}")
                if mixed_precision == 'no':
                    return False
                continue
        return True
    except Exception as e:
        print(f"✗ Accelerator初始化失败: {e}")
        return False

def test_exp_initialization():
    """测试实验类初始化"""
    try:
        args = create_test_args()
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        
        exp = Exp_Long_Term_Forecast(args)
        print("✓ 实验类初始化成功")
        
        if hasattr(exp, 'accelerator') and exp.accelerator is not None:
            print("✓ Accelerate集成成功")
            print(f"  设备: {exp.device}")
        else:
            print("! Accelerate未启用，使用传统训练模式")
            
        return True
    except Exception as e:
        print(f"✗ 实验类初始化失败: {e}")
        return False

def main():
    print("=== Time-Series-Library Accelerate集成测试 ===\n")
    
    tests = [
        ("PyTorch导入测试", test_torch_import),
        ("Accelerate导入测试", test_accelerate_import), 
        ("模型导入测试", test_model_import),
        ("Accelerate初始化测试", test_accelerate_initialization),
        ("实验类初始化测试", test_exp_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"运行 {test_name}...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} 通过\n")
        else:
            print(f"✗ {test_name} 失败\n")
    
    print(f"=== 测试结果: {passed}/{total} 通过 ===")
    
    if passed == total:
        print("🎉 所有测试通过！Accelerate集成正常工作。")
        print("\n可以使用以下命令开始训练:")
        print("accelerate launch --config_file ./accelerate_config.yaml run.py --use_accelerate [其他参数]")
    else:
        print("❌ 部分测试失败，请检查依赖安装和配置。")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
