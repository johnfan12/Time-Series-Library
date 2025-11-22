import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from .hyper_feature_bank import SimpleTSFeature


class Model(nn.Module):
    """DLinear with SimpleTSFeature residuals added to the raw input."""

    def __init__(self, configs, individual: bool = False):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name in ['classification', 'anomaly_detection', 'imputation']:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = getattr(configs, 'individual', individual)
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                seasonal = nn.Linear(self.seq_len, self.pred_len)
                trend = nn.Linear(self.seq_len, self.pred_len)
                seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Seasonal.append(seasonal)
                self.Linear_Trend.append(trend)
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

        feat_dim = self.seq_len * self.channels
        feat_methods = getattr(configs, 'feature_methods', None)
        feat_freqs = getattr(configs, 'feature_n_freqs', 3)
        feat_ar_order = getattr(configs, 'feature_ar_order', 3)
        feat_ar_reg = getattr(configs, 'feature_ar_reg', 1e-4)
        feat_arima_max_p = getattr(configs, 'feature_arima_max_p', 3)
        feat_arima_max_d = getattr(configs, 'feature_arima_max_d', 2)

        self.feat_extractor = SimpleTSFeature(
            seq_len=self.seq_len,
            in_channels=self.channels,
            d_feat=feat_dim,
            method_names=feat_methods,
            n_freqs=feat_freqs,
            ar_order=feat_ar_order,
            ar_regularizer=feat_ar_reg,
            arima_max_p=feat_arima_max_p,
            arima_max_d=feat_arima_max_d,
        )

    def _augment_input(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feat_extractor(x)
        feat = feat.view(x.size(0), self.seq_len, self.channels)
        return x + feat

    def encoder(self, x):
        x = self._augment_input(x)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc)
        return None
