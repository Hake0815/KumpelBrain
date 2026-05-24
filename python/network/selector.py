import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttention, MultiHeadAttentionArgs


class Selector(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_target_inner: int,
        target_attention_args: MultiHeadAttentionArgs,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.selected_marker = nn.Embedding(1, dimension_out, **factory_kwargs)
        self.target_multi_head_attention = MultiHeadAttention.from_args(
            target_attention_args
        )
        self.post_pooling_feed_forward = FeedForward(
            dimension_out, dimension_target_inner, **factory_kwargs
        )
        self.linear_reduce = nn.Linear(dimension_out, 1, **factory_kwargs)
        self.stop_token = nn.Embedding(1, dimension_out, **factory_kwargs)

    def forward(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        partial_selected_cards = transformed_state[2:].index_select(
            0, card_indices.index_select(0, partial_selection)
        )
        candidate_cards = transformed_state[2:].index_select(
            0, card_indices.index_select(0, candidates)
        ) + self.selected_marker(torch.tensor([0]))
        key_values = torch.cat(
            [partial_selected_cards, transformed_state, embedded_interaction.unsqueeze(0)]
        )
        if include_stop_token:
            key_values = torch.cat([key_values, self.stop_token(torch.tensor([0]))])
        pooled = candidate_cards + self.target_multi_head_attention(
            candidate_cards.unsqueeze(0),
            key_values.unsqueeze(0),
            key_values.unsqueeze(0),
        ).squeeze(0)
        pooled = pooled + self.post_pooling_feed_forward(pooled)
        return self.linear_reduce(pooled).squeeze(-1)
