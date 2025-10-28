import torch
import torch.nn as nn
import sys
import os

# Add sundial directory to path to enable absolute imports
sundial_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sundial')
if sundial_path not in sys.path:
    sys.path.insert(0, sundial_path)

# Now import with absolute imports since sundial is in sys.path
import modeling_sundial
import configuration_sundial

# Get the classes we need
SundialForPrediction = modeling_sundial.SundialForPrediction
SundialConfig = configuration_sundial.SundialConfig


class Model(nn.Module):
    """
    Sundial model for time series forecasting
    Paper link: https://arxiv.org/abs/2403.02543
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        
        # Get input_token_len from configs, default to 16 if not specified
        input_token_len = getattr(configs, 'input_token_len', 16)
        
        # Create Sundial configuration
        self.config = SundialConfig(
            input_token_len=input_token_len,
            hidden_size=configs.d_model,
            intermediate_size=configs.d_ff,
            output_token_lens=[configs.pred_len],
            num_hidden_layers=configs.e_layers,
            num_attention_heads=configs.n_heads,
            hidden_act="silu",
            use_cache=False,
            dropout_rate=configs.dropout,
            max_position_embeddings=10000,
            num_sampling_steps=5,  # Reduce from default 50 for faster inference
            flow_loss_depth=2,  # Reduce depth for faster training
            diffusion_batch_mul=2,  # Reduce batch multiplier
        )
        
        # Initialize Sundial model
        self.model = SundialForPrediction(self.config)
        
        # For multivariate forecasting
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Channel projection if needed
        if self.enc_in != 1:
            self.channel_proj = nn.Linear(self.enc_in, 1)
        if self.c_out != 1:
            self.output_proj = nn.Linear(1, self.c_out)
        
        # Simple projection head for fast training (optional fallback)
        self.use_simple_head = False  # Set to True to bypass Sundial inference during training
        if self.use_simple_head:
            self.projection_head = nn.Linear(configs.d_model, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc: [batch_size, seq_len, enc_in]
        Returns: [batch_size, pred_len, c_out]
        """
        batch_size = x_enc.shape[0]
        
        # Handle multivariate input by projecting to univariate or processing each channel
        if self.enc_in > 1:
            # Project channels to single channel
            x_enc = self.channel_proj(x_enc)  # [batch_size, seq_len, 1]
        
        # Reshape to [batch_size, seq_len] for Sundial input
        x_enc = x_enc.squeeze(-1)  # [batch_size, seq_len]
        
        # Use the model's forward method for inference
        # Pass labels=None to trigger inference mode in SundialForPrediction
        outputs = self.model(
            input_ids=x_enc,
            labels=None,  # No labels means inference mode
            max_output_length=self.pred_len,
            revin=True,  # Use reversible instance normalization
            num_samples=1,
        )
        
        # Get predictions from the model output
        # outputs.logits contains the predictions: [batch_size, num_samples, pred_len]
        predictions = outputs.logits

        # Ensure predictions has correct shape
        if predictions is None:
            # Fallback: create dummy predictions with correct shape
            predictions = torch.zeros(batch_size, self.pred_len, 1, device=x_enc.device)
        else:
            # Handle different possible shapes
            # Expected: [batch_size, num_samples, pred_len] where num_samples=1
            # Or: [batch_size, pred_len]
            
            while len(predictions.shape) > 2:
                # Keep removing dimensions until we get [batch_size, pred_len]
                if predictions.shape[1] == 1:
                    predictions = predictions.squeeze(1)
                else:
                    # If num_samples > 1, take the first sample
                    predictions = predictions[:, 0, :]
            
            # Now predictions should be [batch_size, pred_len]
            # Add channel dimension: [batch_size, pred_len] -> [batch_size, pred_len, 1]
            predictions = predictions.unsqueeze(-1)
        
        # Verify shape before projection
        assert predictions.shape[0] == batch_size, f"Batch size mismatch: {predictions.shape[0]} vs {batch_size}"
        assert predictions.shape[1] == self.pred_len, f"Pred len mismatch: {predictions.shape[1]} vs {self.pred_len}"
        
        # Project back to output channels if needed
        if self.c_out > 1:
            predictions = self.output_proj(predictions)  # [batch_size, pred_len, c_out]
        
        return predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, L, D]
        return None
