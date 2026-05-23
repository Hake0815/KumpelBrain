"""Save/load roundtrip tests for embedding holders and GameEmbedding."""

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

import card_embedding_forward_fixtures as card_fixtures  # noqa: E402
import card_state_embedding_forward_fixtures as card_state_fixtures  # noqa: E402
import game_embedding_fixtures as game_fixtures  # noqa: E402
import kumpel_embedding  # noqa: E402
from golden_test_utils import deterministic_algorithms, seed_for_device  # noqa: E402

DIM = card_fixtures.FIXTURE_DIMENSION_OUT


@pytest.fixture(params=["cpu"])
def device(request) -> torch.device:
    return torch.device(request.param)


def test_card_embedding_holder_save_load_smoke(device: torch.device, tmp_path) -> None:
    """Smoke test for standalone make_card_embedding persistence.

    CardEmbedding is a test/debug helper: SharedEmbeddingHolder is caller-owned and not
    registered on the holder. Use GameEmbedding.save_weights for full-model persistence.
  """
    model = kumpel_embedding.make_card_embedding(None, DIM, device=device, dtype=torch.float32)
    cards = card_fixtures.FIXTURE_CASES["single_variant_0"]
    with torch.inference_mode():
        model.forward(cards)

    path = tmp_path / "card_embedding.weights"
    model.save_weights(str(path))
    assert path.stat().st_size > 0
    model.load_weights(str(path))


def test_card_state_embedding_holder_save_load_roundtrip(device: torch.device, tmp_path) -> None:
    model_a = kumpel_embedding.make_card_state_embedding(DIM, device=device, dtype=torch.float32)
    states = card_state_fixtures.FIXTURE_CASES["single_no_relations"]
    with torch.inference_mode():
        out_a, idx_a = model_a.forward(states)

    path = tmp_path / "card_state_embedding.weights"
    model_a.save_weights(str(path))

    model_b = kumpel_embedding.make_card_state_embedding(DIM, device=device, dtype=torch.float32)
    model_b.load_weights(str(path))
    with torch.inference_mode():
        out_b, idx_b = model_b.forward(states)

    torch.testing.assert_close(out_a, out_b)
    torch.testing.assert_close(idx_a, idx_b)


def test_game_embedding_save_load_roundtrip(device: torch.device, tmp_path) -> None:
    with deterministic_algorithms(True):
        seed_for_device(device)
        model_a = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
        model_a.eval()
        game_state = game_fixtures.EMBED_GAME_STATE_CASES["three_cards"]
        interactions = game_fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"][1]

        with torch.inference_mode():
            state_a, idx_a = model_a.embedGameState(game_state)
            cards_a = game_fixtures.extract_card_embeddings(state_a)
            inter_a = model_a.embedGameInteraction(interactions, idx_a, cards_a)

        path = tmp_path / "game_embedding.weights"
        model_a.save_weights(str(path))

        seed_for_device(device)
        model_b = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
        model_b.load_weights(str(path))
        model_b.eval()

        with torch.inference_mode():
            state_b, idx_b = model_b.embedGameState(game_state)
            cards_b = game_fixtures.extract_card_embeddings(state_b)
            inter_b = model_b.embedGameInteraction(interactions, idx_b, cards_b)

    torch.testing.assert_close(state_a, state_b)
    torch.testing.assert_close(idx_a, idx_b)
    torch.testing.assert_close(inter_a, inter_b)
