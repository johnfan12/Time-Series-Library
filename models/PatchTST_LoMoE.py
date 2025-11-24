import torch

from layers.LoMoE_Layers import LoMoEOutputHead
from models.PatchTST import FlattenHead, Model as PatchTSTBase


class Model(PatchTSTBase):
    """PatchTST variant with a lightweight MoE output layer."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(configs, patch_len=patch_len, stride=stride)
        self._router_probs = None
        self._lomoe_active = self.task_name in {'long_term_forecast', 'short_term_forecast'}
        if self._lomoe_active:
            if not isinstance(self.head, FlattenHead):
                raise ValueError('LoMoE head expects a FlattenHead for forecasting tasks.')
            num_experts = getattr(configs, 'num_experts', 4)
            rank = getattr(configs, 'lora_rank', 8)
            top_k = getattr(configs, 'moe_topk', 2)
            self.head = LoMoEOutputHead(
                base_head=self.head,
                d_model=configs.d_model,
                num_experts=num_experts,
                rank=rank,
                top_k=top_k,
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self._lomoe_active:
            head_out, router_probs = self.head(enc_out)
            self._router_probs = router_probs
        else:
            head_out = self.head(enc_out)
            self._router_probs = None
        dec_out = head_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            dec_out = dec_out[:, -self.pred_len:, :]
            if self._lomoe_active:
                return dec_out, self._router_probs
            return dec_out
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    def get_router_probs(self):
        return self._router_probs
