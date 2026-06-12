from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin

if TYPE_CHECKING:
    from profiling import InferenceProfiler
from state_transformer import StateTransformer
from kumpel_embedding import GameEmbedding
from game_embedding import extract_card_embeddings
from interaction_network import InteractionNetwork
from multi_head_attention import MultiHeadAttentionArgs


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor if tensor.device == device else tensor.to(device)


class KumpelNetwork(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_state_inner: int,
        dimension_interaction_inner: int,
        state_attention_args: MultiHeadAttentionArgs,
        interaction_attention_args: MultiHeadAttentionArgs,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        embedding_device: torch.device | None = None,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding_device = embedding_device or torch.device("cpu")
        super().__init__()
        self.state_transformer = StateTransformer(
            dimension_out,
            dimension_state_inner,
            state_attention_args,
            num_layers,
            **self.factory_kwargs
        )
        self.game_embedding = GameEmbedding(
            dimension_out, self.embedding_device, dtype
        )
        self.interaction_network = InteractionNetwork(
            dimension_out,
            dimension_interaction_inner,
            interaction_attention_args,
            **self.factory_kwargs
        )
        self.profiler: InferenceProfiler | None = None

    def forward_batch(
        self,
        game_states: list[bytes],
        interactions_per_game: list[list[bytes]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        compute_device = self.factory_kwargs["device"]
        embedding_device = self.embedding_device
        profiler = self.profiler

        if profiler is not None:
            with profiler.timed(embedding_device) as span:
                embedded_game_state, state_mask, card_indices = (
                    self.game_embedding.embedGameState(game_states)
                )
            profiler.embed_state_s += span.elapsed
            profiler.forward_calls += 1

            with profiler.timed(compute_device) as span:
                embedded_game_state = _to_device(embedded_game_state, compute_device)
                state_mask = _to_device(state_mask, compute_device)
            profiler.to_compute_s += span.elapsed

            with profiler.timed(compute_device) as span:
                transformed_state = self.state_transformer(
                    embedded_game_state, key_padding_mask=state_mask
                )
            profiler.state_transformer_s += span.elapsed

            with profiler.timed(compute_device) as span:
                transformed_cards = _to_device(
                    extract_card_embeddings(transformed_state), embedding_device
                )
            profiler.cards_to_cpu_s += span.elapsed

            with profiler.timed(embedding_device) as span:
                embedded_interactions, int_mask = (
                    self.game_embedding.embedGameInteraction(
                        interactions_per_game, card_indices, transformed_cards
                    )
                )
                embedded_interactions = _to_device(
                    embedded_interactions, compute_device
                )
                int_mask = _to_device(int_mask, compute_device)
            profiler.embed_interactions_s += span.elapsed

            with profiler.timed(compute_device) as span:
                interaction_scores = self.interaction_network(
                    embedded_interactions, transformed_state, key_mask=state_mask
                )
                interaction_scores = interaction_scores.masked_fill(~int_mask, float("-inf"))
            profiler.interaction_network_s += span.elapsed
        else:
            embedded_game_state, state_mask, card_indices = (
                self.game_embedding.embedGameState(game_states)
            )
            embedded_game_state = _to_device(embedded_game_state, compute_device)
            state_mask = _to_device(state_mask, compute_device)

            transformed_state = self.state_transformer(
                embedded_game_state, key_padding_mask=state_mask
            )
            transformed_cards = _to_device(
                extract_card_embeddings(transformed_state), embedding_device
            )

            embedded_interactions, int_mask = self.game_embedding.embedGameInteraction(
                interactions_per_game, card_indices, transformed_cards
            )
            embedded_interactions = _to_device(embedded_interactions, compute_device)
            int_mask = _to_device(int_mask, compute_device)

            interaction_scores = self.interaction_network(
                embedded_interactions, transformed_state, key_mask=state_mask
            )
            interaction_scores = interaction_scores.masked_fill(~int_mask, float("-inf"))

        return (
            interaction_scores,
            transformed_state,
            embedded_interactions,
            _to_device(card_indices, compute_device),
            int_mask,
            state_mask,
        )

    def forward(
        self, game_state: bytes, game_interactions: list[bytes]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            interaction_scores,
            transformed_state,
            embedded_interactions,
            card_indices,
            int_mask,
            state_mask,
        ) = self.forward_batch([game_state], [game_interactions])

        valid_state = state_mask[0]
        valid_int = int_mask[0]

        return (
            interaction_scores[0, valid_int],
            transformed_state[0, valid_state],
            embedded_interactions[0, valid_int],
            card_indices[0],
        )
