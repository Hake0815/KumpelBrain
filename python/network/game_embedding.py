"""Helpers for kumpel_embedding.GameEmbedding training and inference."""

from __future__ import annotations

import torch

NUM_PLAYER_STATE_ROWS = 2


def extract_card_embeddings(game_state_embedding: torch.Tensor) -> torch.Tensor:
    """Return card rows from a game-state embedding.

  ``embedGameState`` returns shape ``[B, 2 + max_cards, dim]`` (or ``[2 + num_cards, dim]`` for a
  single 2D state). Player rows are stripped so the result is ``[B, max_cards, dim]`` or
  ``[num_cards, dim]``.
    """
    if game_state_embedding.dim() == 3:
        return game_state_embedding[:, NUM_PLAYER_STATE_ROWS:, :]
    return game_state_embedding[NUM_PLAYER_STATE_ROWS:]
