import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerLayer(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_inner: int,
        dimension_head: int,
        nheads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(dimension_out, dimension_out, dimension_out, dimension_head, nheads, dropout, bias, **factory_kwargs)
        self.feed_forward = FeedForward(dimension_out, dimension_inner, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.multi_head_attention(x))