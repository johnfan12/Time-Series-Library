from typing import Optional

import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from .HyperTS import SimpleTSFeature, HyperLinear


class PatchTSTBackbone(nn.Module):
    """PatchTST-style encoder used as shared backbone."""

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        factor: int,
        dropout: float,
        activation: str,
        patch_len: int,
        stride: int,
    ) -> None:
        super().__init__()
        padding = stride
        self.d_model = d_model
        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=dropout,
        )
        attn_layers = [
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
            for _ in range(n_layers)
        ]
        self.encoder = Encoder(attn_layers, norm_layer=nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        means = x.mean(dim=1, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_centered / stdev

        x_perm = x_norm.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_perm)
        enc_out, _ = self.encoder(enc_out)
        patch_steps = enc_out.shape[1]
        enc_out = enc_out.view(B, n_vars, patch_steps, self.d_model)
        latent = enc_out.mean(dim=(1, 2))
        return latent


class Model(nn.Module):
    """HyperTS variant with PatchTST backbone and hypernetwork head."""

    def __init__(self, configs) -> None:
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

        patch_len = getattr(configs, "patch_len", 16)
        stride = getattr(configs, "stride", 8)
        n_heads = getattr(configs, "n_heads", 4)
        n_layers = getattr(configs, "e_layers", 2)
        d_ff = getattr(configs, "d_ff", d_model * 4)
        factor = getattr(configs, "factor", 3)
        dropout = getattr(configs, "dropout", 0.1)
        activation = getattr(configs, "activation", "gelu")

        self.feat_extractor = SimpleTSFeature(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_feat=d_feat,
        )

        self.backbone = PatchTSTBackbone(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            activation=activation,
            patch_len=patch_len,
            stride=stride,
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
            f"Task {self.task_name} is not implemented for HyperTSPatch."
        )
