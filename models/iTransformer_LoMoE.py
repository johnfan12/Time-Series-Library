import torch

from layers.LoMoE_Layers import LoMoELinearHead
from models.iTransformer import Model as ITransformerBase


class Model(ITransformerBase):
    """iTransformer variant equipped with the shared LoMoE head."""

    def __init__(self, configs):
        super().__init__(configs)
        self._router_probs = None
        self._lomoe_active = self.task_name in {'long_term_forecast', 'short_term_forecast'}
        if self._lomoe_active:
            num_experts = getattr(configs, 'num_experts', 4)
            rank = getattr(configs, 'lora_rank', 8)
            top_k = getattr(configs, 'moe_topk', 2)
            self.projection = LoMoELinearHead(
                base_linear=self.projection,
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

        _, _, n_vars = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        if self._lomoe_active:
            proj_out, router_probs = self.projection(enc_out)
            self._router_probs = router_probs
        else:
            proj_out = self.projection(enc_out)
            self._router_probs = None

        dec_out = proj_out.permute(0, 2, 1)[:, :, :n_vars]
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
