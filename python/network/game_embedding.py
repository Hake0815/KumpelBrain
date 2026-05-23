"""Helpers for kumpel_embedding.GameEmbedding training and inference."""

from __future__ import annotations

import torch

NUM_PLAYER_STATE_ROWS = 2


def extract_card_embeddings(game_state_embedding: torch.Tensor) -> torch.Tensor:
    """Return card rows from a game-state embedding.

    ``embedGameState`` returns shape ``[2 + num_cards, dim]`` where the first two rows are
    player states and the remainder are card embeddings in batch order.
    """
    return game_state_embedding[NUM_PLAYER_STATE_ROWS:]
