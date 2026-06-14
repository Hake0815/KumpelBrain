"""Shared pre-norm cross-attention scoring block for interaction and target heads."""

import torch
import torch.nn as nn
from action_scores import ActionScores
from save_load_mixin import SaveLoadMixin
from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention, MultiHeadAttentionArgs


class CrossAttentionScoringBlock(nn.Module, SaveLoadMixin):
    """Pre-norm block: optional FFN, cross-attention, FFN, scalar head.

    Callers pass batched sequences (``InteractionNetwork``, ``Selector`` use ``N=1``).
    Feature width is ``dimension_out`` (``d_q`` in ``attention_args``).
    """

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
        self.value_head = nn.Linear(dimension_out, 1, **factory_kwargs)
        self.policy_head = nn.Linear(dimension_out, 1, **factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
    ) -> ActionScores:
        # query: (N, L_q, D) — one row per item to score (e.g. interactions or target candidates)
        # key, value: (N, L_kv, D) — context attended over (e.g. full state, or state + selection context)
        # Returns two (N, L_q) logits: calibrated value and action policy.
        x = query  # (N, L_q, D)
        if self.include_pre_ffn:
            x = x + self.first_feed_forward(self.norm_pre_ffn(x))  # (N, L_q, D)
        x = x + self.multi_head_attention(
            self.norm_attention(x), key, value, attn_mask=attn_mask
        )  # (N, L_q, D)
        x = x + self.post_feed_forward(self.norm_post_ffn(x))  # (N, L_q, D)
        return ActionScores(
            value_logits=self.value_head(x).squeeze(-1),
            policy_logits=self.policy_head(x).squeeze(-1),
        )
