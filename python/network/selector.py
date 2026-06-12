import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from game_embedding import extract_card_embeddings
from multi_head_attention import MultiHeadAttentionArgs
from scoring_block import CrossAttentionScoringBlock


class Selector(nn.Module, SaveLoadMixin):
    """Score target candidates via cross-attention over selection context and state.

    Feature width is ``dimension_out`` (``D``). ``forward`` batches a single game
    (``N=1``) for ``CrossAttentionScoringBlock``.
    """

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
            torch.zeros(1, dtype=torch.long, device=device),
            persistent=False,
        )

    def _build_candidate_and_context(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if card_indices.dim() == 2:
            card_indices = card_indices.squeeze(0)
        if transformed_state.dim() == 3:
            transformed_state = transformed_state.squeeze(0)
        device = transformed_state.device
        candidates = candidates.to(device)
        partial_selection = partial_selection.to(device)
        card_indices = card_indices.to(device)
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
        return candidate_cards, key_values

    def forward(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        # candidates: (L_c,) int64 — deck_ids of rows to score
        # partial_selection: (L_p,) int64 — already chosen targets (may be empty)
        # transformed_state: (L_state, D) — player rows + card rows from state transformer
        # embedded_interaction: (D,) — one interaction embedding
        # card_indices: (L,) or (1, L) int64 — maps deck_id -> row index in card_rows
        # Returns: (L_c,) — one score per candidate (scoring_block runs with N=1)
        candidate_cards, key_values = self._build_candidate_and_context(
            candidates,
            partial_selection,
            transformed_state,
            embedded_interaction,
            card_indices,
            include_stop_token,
        )
        key_values_batch = key_values.unsqueeze(0)
        return self.scoring_block(
            candidate_cards.unsqueeze(0),
            key_values_batch,
            key_values_batch,
        ).squeeze(0)

    def forward_batch(
        self,
        candidates_per_game: list[torch.Tensor],
        partial_selection_per_game: list[torch.Tensor],
        transformed_state_per_game: list[torch.Tensor],
        embedded_interaction_per_game: list[torch.Tensor],
        card_indices_per_game: list[torch.Tensor],
        include_stop_token_per_game: list[bool],
    ) -> torch.Tensor:
        batch_size = len(candidates_per_game)
        if batch_size == 0:
            return torch.empty(0, 0)

        dim = transformed_state_per_game[0].size(-1)
        device = transformed_state_per_game[0].device
        dtype = transformed_state_per_game[0].dtype

        candidate_cards_list: list[torch.Tensor] = []
        key_values_list: list[torch.Tensor] = []
        candidate_counts: list[int] = []

        for i in range(batch_size):
            candidate_cards, key_values = self._build_candidate_and_context(
                candidates_per_game[i],
                partial_selection_per_game[i],
                transformed_state_per_game[i],
                embedded_interaction_per_game[i],
                card_indices_per_game[i],
                include_stop_token_per_game[i],
            )
            candidate_cards_list.append(candidate_cards)
            key_values_list.append(key_values)
            candidate_counts.append(candidate_cards.size(0))

        max_candidates = max(candidate_counts)
        max_context = max(kv.size(0) for kv in key_values_list)

        padded_candidates = torch.zeros(
            batch_size, max_candidates, dim, device=device, dtype=dtype
        )
        padded_key_values = torch.zeros(
            batch_size, max_context, dim, device=device, dtype=dtype
        )
        candidate_mask = torch.zeros(
            batch_size, max_candidates, device=device, dtype=torch.bool
        )
        key_mask = torch.zeros(batch_size, max_context, device=device, dtype=torch.bool)

        for i in range(batch_size):
            n_c = candidate_counts[i]
            n_kv = key_values_list[i].size(0)
            padded_candidates[i, :n_c] = candidate_cards_list[i]
            padded_key_values[i, :n_kv] = key_values_list[i]
            candidate_mask[i, :n_c] = True
            key_mask[i, :n_kv] = True

        attn_mask = key_mask.unsqueeze(1).expand(batch_size, max_candidates, max_context)
        scores = self.scoring_block(
            padded_candidates, padded_key_values, padded_key_values, attn_mask=attn_mask
        )
        return scores.masked_fill(~candidate_mask, float("-inf"))
