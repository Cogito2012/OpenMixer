import torch
import torch.nn as nn
from collections import OrderedDict



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop: float = 0., return_kv=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=drop)
        self.ln_x = LayerNorm(d_model)
        self.ln_y = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("fc_drop", nn.Dropout(drop)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("proj_drop", nn.Dropout(drop)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.return_kv = return_kv

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ x: query (T=1, B, d)
            y: key & value (T=64, B, d)
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        x = x + self.attention(self.ln_x(x), self.ln_y(y))
        x = x + self.mlp(self.ln_2(x))
        if x.size(0) == 1:
            x = x.squeeze(0)
        if self.return_kv:
            return x, y
        return x


class CrossAttnModules(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input[0]