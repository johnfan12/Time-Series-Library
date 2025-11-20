from typing import Optional, Tuple

import torch
import torch.nn as nn


class SimpleTSFeature(nn.Module):
    """Extracts lightweight statistical features from a window."""

    def __init__(self, seq_len: int, in_channels: int, d_feat: int, hidden: Optional[int] = None):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        base_dim = in_channels * 5
        if hidden is None:
            hidden = max(base_dim, d_feat)

        self.proj = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_feat),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: [B, d_feat]
        """
        mean = x.mean(dim=1)
        var = x.var(dim=1, unbiased=False)
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        last = x[:, -1, :]
        if x.shape[1] > 1:
            diff = x[:, 1:, :] - x[:, :-1, :]
            diff_mean = diff.mean(dim=1)
            autocorr = (x[:, 1:, :] * x[:, :-1, :]).mean(dim=1)
        else:
            diff_mean = torch.zeros_like(mean)
            autocorr = torch.zeros_like(mean)

        stats = torch.cat([mean, std, last, diff_mean, autocorr], dim=-1)
        return self.proj(stats)


class SimpleBackbone(nn.Module):
    """Simple MLP backbone shared across all windows."""

    def __init__(self, seq_len: int, in_channels: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        input_dim = seq_len * in_channels

        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x_flat = x.reshape(B, L * C)
        return self.net(x_flat)


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
    """HyperTS forecasting model with a shared MLP backbone and hypernetwork head."""

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

        self.feat_extractor = SimpleTSFeature(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_feat=d_feat,
        )

        self.backbone = SimpleBackbone(
            seq_len=self.seq_len,
            in_channels=self.c_in,
            d_model=d_model,
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
