"""Non-golden smoke tests for additional GameEmbedding interaction scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"

for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import game_embedding_fixtures as fixtures  # noqa: E402
import kumpel_embedding  # noqa: E402

DIM = fixtures.FIXTURE_DIMENSION_OUT


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.mark.parametrize("case_id", sorted(fixtures.EXTENDED_GAME_INTERACTION_CASES.keys()))
def test_game_embedding_extended_interaction_smoke(case_id: str, device: torch.device) -> None:
    game_state_bytes, interaction_bytes = fixtures.EXTENDED_GAME_INTERACTION_CASES[case_id]
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        state_emb, _mask, indices = model.embedGameState([game_state_bytes])
        cards = fixtures.extract_card_embeddings(state_emb)
        out, _interaction_mask = model.embedGameInteraction([interaction_bytes], indices, cards)
    out = out[0]
    assert out.ndim == 2
    assert out.shape[1] == DIM
    assert out.shape[0] == len(interaction_bytes)
    assert torch.isfinite(out).all()


def test_game_embedding_sparse_deck_state_smoke(device: torch.device) -> None:
    payload = fixtures.EXTENDED_GAME_STATE_CASES["sparse_deck_ids"]
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        embedding, _mask, card_indices = model.embedGameState([payload])
    assert embedding.shape == (1, 2 + 3, DIM)
    assert card_indices.shape[1] >= 13
    assert torch.isfinite(embedding).all()
