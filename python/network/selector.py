import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from game_embedding import extract_card_embeddings
from multi_head_attention import MultiHeadAttentionArgs
from scoring_block import CrossAttentionScoringBlock


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
        self.scoring_block = CrossAttentionScoringBlock(
            dimension_out,
            dimension_target_inner,
            target_attention_args,
            include_pre_ffn=False,
            **factory_kwargs,
        )
        self.stop_token = nn.Embedding(1, dimension_out, **factory_kwargs)
        self.register_buffer(
            "_embed_index",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        card_rows = extract_card_embeddings(transformed_state)
        partial_selected_cards = card_rows.index_select(
            0, card_indices.index_select(0, partial_selection)
        )
        candidate_cards = card_rows.index_select(
            0, card_indices.index_select(0, candidates)
        ) + self.selected_marker(self._embed_index)
        key_values = torch.cat(
            [
                partial_selected_cards,
                transformed_state,
                embedded_interaction.unsqueeze(0),
            ]
        )
        if include_stop_token:
            key_values = torch.cat([key_values, self.stop_token(self._embed_index)])
        key_values_batch = key_values.unsqueeze(0)
        return self.scoring_block(
            candidate_cards.unsqueeze(0),
            key_values_batch,
            key_values_batch,
        )
