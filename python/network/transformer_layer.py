import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from multi_head_attention import MultiHeadAttention, MultiHeadAttentionArgs
from feed_forward import FeedForward


class TransformerLayer(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_inner: int,
        attention_args: MultiHeadAttentionArgs,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_attention = nn.LayerNorm(dimension_out, **factory_kwargs)
        self.multi_head_attention = MultiHeadAttention.from_args(attention_args)
        self.norm_feed_forward = nn.LayerNorm(dimension_out, **factory_kwargs)
        self.feed_forward = FeedForward(
            dimension_out, dimension_inner, **factory_kwargs
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        normed = self.norm_attention(x)
        x = x + self.multi_head_attention(normed, normed, normed, attn_mask=attn_mask)
        x = x + self.feed_forward(self.norm_feed_forward(x))
        return x
