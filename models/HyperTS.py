import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .hyper_feature_bank import SimpleTSFeature


class PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self._pos_table: torch.Tensor
        self.register_buffer('_pos_table', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pos_table[:, :x.size(1), :]


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone shared across all windows."""

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        h = self.input_proj(x)
        h = h + self.positional_encoding(h)
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(dim=1)


class HyperLinear(nn.Module):
    """Generates linear-layer weights from a conditioning vector."""

    def __init__(self, d_feat: int, latent_dim: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        if hidden is None:
            hidden = max(d_feat, latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(d_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.fc_w = nn.Linear(hidden, out_dim * latent_dim)
        self.fc_b = nn.Linear(hidden, out_dim)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(feat)
        B = h.shape[0]
        W = self.fc_w(h).view(B, self.out_dim, self.latent_dim)
        b = self.fc_b(h)
        return W, b


class Model(nn.Module):
    """HyperTS forecasting model with a Transformer backbone and hypernetwork head."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, "label_len", 0)
        self.c_in = configs.enc_in
        self.c_out = configs.c_out

        if self.task_name in ["classification", "anomaly_detection", "imputation"]:
            self.pred_len = self.seq_len
        else:
            self.pred_len = configs.pred_len

        d_model = configs.d_model
        d_feat = d_model
        out_dim = self.pred_len * self.c_out

        feat_freqs = getattr(configs, "feature_n_freqs", 3)
        feat_ar_order = getattr(configs, "feature_ar_order", 3)
        feat_ar_reg = getattr(configs, "feature_ar_reg", 1e-4)
        feat_methods = getattr(configs, "feature_methods", None)

        self.feat_extractor = SimpleTSFeature(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_feat=d_feat,
            method_names=feat_methods,
            n_freqs=feat_freqs,
            ar_order=feat_ar_order,
            ar_regularizer=feat_ar_reg,
        )

        n_heads = getattr(configs, "n_heads", 4)
        n_layers = getattr(configs, "e_layers", 2)
        d_ff = getattr(configs, "d_ff", d_model * 4)
        dropout = getattr(configs, "dropout", 0.1)

        self.backbone = TransformerBackbone(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.hyper = HyperLinear(
            d_feat=d_feat,
            latent_dim=d_model,
            out_dim=out_dim,
        )

    def forecast(self, x_enc: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        feat = self.feat_extractor(x_enc)
        h = self.backbone(x_enc)
        W, b = self.hyper(feat)

        h_vec = h.unsqueeze(-1)
        y_flat = torch.bmm(W, h_vec).squeeze(-1) + b
        B = x_enc.shape[0]
        return y_flat.view(B, self.pred_len, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        raise NotImplementedError(
            f"Task {self.task_name} is not implemented for HyperTS."
        )
