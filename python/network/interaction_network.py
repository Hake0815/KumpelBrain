import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention, MultiHeadAttentionArgs


class InteractionNetwork(nn.Module, SaveLoadMixin):
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
        self.first_feed_forward = FeedForward(
            dimension_out, dimension_inner, **factory_kwargs
        )
        self.multi_head_attention = MultiHeadAttention.from_args(attention_args)
        self.post_pooling_feed_forward = FeedForward(dimension_out, dimension_inner, **factory_kwargs)
        self.linear_reduce = nn.Linear(dimension_out, 1, **factory_kwargs)

    def forward(self, embedded_interactions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = embedded_interactions + self.first_feed_forward(embedded_interactions)
        x = x + self.multi_head_attention(x, state, state)
        x = x + self.post_pooling_feed_forward(x)
        return self.linear_reduce(x).squeeze(-1)
