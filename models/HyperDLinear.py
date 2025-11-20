from typing import Optional

import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from .HyperTS import SimpleTSFeature, HyperLinear


class DLinearBackbone(nn.Module):
    """DLinear-style backbone that produces a latent vector for Hyper networks."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        moving_avg: int,
        d_model: int,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        self.decomposition = series_decomp(moving_avg)

        if self.individual:
            self.seasonal_layers = nn.ModuleList()
            self.trend_layers = nn.ModuleList()
            for _ in range(self.channels):
                seasonal = nn.Linear(self.seq_len, self.pred_len)
                trend = nn.Linear(self.seq_len, self.pred_len)
                seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.seasonal_layers.append(seasonal)
                self.trend_layers.append(trend)
        else:
            self.seasonal_linear = nn.Linear(self.seq_len, self.pred_len)
            self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
            self.seasonal_linear.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.trend_linear.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

        self.proj = nn.Linear(self.channels * self.pred_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            device = seasonal_init.device
            dtype = seasonal_init.dtype
            seasonal_output = torch.zeros(
                seasonal_init.size(0), seasonal_init.size(1), self.pred_len, device=device, dtype=dtype
            )
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.seasonal_layers[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.trend_layers[i](trend_init[:, i, :])
        else:
            seasonal_output = self.seasonal_linear(seasonal_init)
            trend_output = self.trend_linear(trend_init)
        out = seasonal_output + trend_output
        out = out.permute(0, 2, 1)  # [B, pred_len, C]
        latent = self.proj(out.reshape(out.size(0), -1))
        return latent


class Model(nn.Module):
    """HyperDLinear: HyperTS head on top of a DLinear backbone."""

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
        moving_avg = getattr(configs, "moving_avg", 25)
        individual = getattr(configs, "individual", False)

        self.feat_extractor = SimpleTSFeature(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_feat=d_feat,
        )

        self.backbone = DLinearBackbone(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.c_in,
            moving_avg=moving_avg,
            d_model=d_model,
            individual=individual,
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
            f"Task {self.task_name} is not implemented for HyperDLinear."
        )
