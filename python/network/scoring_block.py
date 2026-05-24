"""Shared pre-norm cross-attention scoring block for interaction and target heads."""

import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention, MultiHeadAttentionArgs


class CrossAttentionScoringBlock(nn.Module, SaveLoadMixin):
    """Pre-norm block: optional FFN, cross-attention, FFN, scalar head."""

    def __init__(
        self,
        dimension_out: int,
        dimension_inner: int,
        attention_args: MultiHeadAttentionArgs,
        *,
        include_pre_ffn: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.include_pre_ffn = include_pre_ffn
        if include_pre_ffn:
            self.norm_pre_ffn = nn.LayerNorm(dimension_out, **factory_kwargs)
            self.first_feed_forward = FeedForward(
                dimension_out, dimension_inner, **factory_kwargs
            )
        self.norm_attention = nn.LayerNorm(dimension_out, **factory_kwargs)
        self.multi_head_attention = MultiHeadAttention.from_args(attention_args)
        self.norm_post_ffn = nn.LayerNorm(dimension_out, **factory_kwargs)
        self.post_feed_forward = FeedForward(
            dimension_out, dimension_inner, **factory_kwargs
        )
        self.linear_reduce = nn.Linear(dimension_out, 1, **factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        x = query
        if self.include_pre_ffn:
            x = x + self.first_feed_forward(self.norm_pre_ffn(x))
        x = x + self.multi_head_attention(
            self.norm_attention(x), key, value
        )
        x = x + self.post_feed_forward(self.norm_post_ffn(x))
        return self.linear_reduce(x).squeeze(-1)
