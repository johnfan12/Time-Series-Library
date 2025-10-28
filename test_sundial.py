import torch
import sys
import argparse

# Create a simple config object
class Config:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.pred_len = 96
        self.label_len = 48
        self.enc_in = 7
        self.c_out = 7
        self.d_model = 64
        self.d_ff = 128
        self.e_layers = 2
        self.n_heads = 4
        self.dropout = 0.1
        self.input_token_len = 16

# Test the model
try:
    from models.Sundial import Model
    
    print("✓ Model imported successfully")
    
    # Create config
    config = Config()
    
    # Initialize model
    model = Model(config)
    print("✓ Model initialized successfully")
    
    # Create dummy input
    batch_size = 2
    x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
    x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # time features
    x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.c_out)
    x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
    
    print(f"✓ Input shapes:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_mark_enc: {x_mark_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    print(f"  x_mark_dec: {x_mark_dec.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [{batch_size}, {config.pred_len}, {config.c_out}]")
    
    if output.shape == (batch_size, config.pred_len, config.c_out):
        print("\n✅ All tests passed! Sundial model is working correctly.")
    else:
        print(f"\n⚠️ Output shape mismatch!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
