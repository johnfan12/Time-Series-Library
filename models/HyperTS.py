from typing import Optional, Tuple

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer

from .hyper_feature_bank import SimpleTSFeature


class TransformerBackbone(nn.Module):
    """Matches the Transformer encoder stack used by the vanilla Transformer model."""

    def __init__(self, configs):
        super().__init__()
        d_model = configs.d_model
        dropout = getattr(configs, "dropout", 0.1)
        embed_type = getattr(configs, "embed", "fixed")
        freq = getattr(configs, "freq", "h")
        factor = getattr(configs, "factor", 5)
        activation = getattr(configs, "activation", "gelu")
        n_heads = getattr(configs, "n_heads", 4)
        d_ff = getattr(configs, "d_ff", d_model * 4)
        e_layers = getattr(configs, "e_layers", 2)

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            d_model,
            embed_type=embed_type,
            freq=freq,
            dropout=dropout,
        )

        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        False,
                        factor,
                        attention_dropout=dropout,
                        output_attention=False,
                    ),
                    d_model,
                    n_heads,
                ),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(e_layers)
        ]

        self.encoder = Encoder(
            encoder_layers,
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out


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
        feat_arima_max_p = getattr(configs, "feature_arima_max_p", 3)
        feat_arima_max_d = getattr(configs, "feature_arima_max_d", 2)
        feat_methods = getattr(configs, "feature_methods", None)

        self.feat_extractor = SimpleTSFeature(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_feat=d_feat,
            method_names=feat_methods,
            n_freqs=feat_freqs,
            ar_order=feat_ar_order,
            ar_regularizer=feat_ar_reg,
            arima_max_p=feat_arima_max_p,
            arima_max_d=feat_arima_max_d,
        )

        self.backbone = TransformerBackbone(configs)

        self.hyper = HyperLinear(
            d_feat=d_feat,
            latent_dim=d_model,
            out_dim=out_dim,
        )

    def forecast(self, x_enc: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None) -> torch.Tensor:
        feat = self.feat_extractor(x_enc)
        h_seq = self.backbone(x_enc, x_mark_enc)
        h = h_seq.mean(dim=1)
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
