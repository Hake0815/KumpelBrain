"""Deck-id validation for GameEmbedding.embedGameInteraction."""

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
import proto_serialization  # noqa: E402

DIM = fixtures.FIXTURE_DIMENSION_OUT


def _make_target_interaction(target_deck_ids: list[int]) -> bytes:
    pb2 = proto_serialization._load_proto_module()
    interaction = pb2.ProtoBufGameInteraction()
    interaction.type = pb2.GAME_INTERACTION_TYPE_SELECT_CARDS
    data = interaction.data.add()
    data.data_type = pb2.GAME_INTERACTION_DATA_TYPE_TARGET_DATA
    target = data.target_data
    target.possible_targets.extend(target_deck_ids)
    target.target_action = pb2.ACTION_ON_SELECTION_DISCARD
    target.remainder_action = pb2.ACTION_ON_SELECTION_TAKE_TO_HAND
    target.number_of_targets = 1
    return interaction.SerializeToString()


def test_embed_game_interaction_rejects_missing_deck_id() -> None:
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    game_state = fixtures.EMBED_GAME_STATE_CASES["three_cards"]
    with torch.inference_mode():
        state_emb, _mask, indices = model.embedGameState([game_state])
        cards = fixtures.extract_card_embeddings(state_emb)

    bad_interaction = [_make_target_interaction([99])]
    with pytest.raises((RuntimeError, ValueError), match="deck_id"):
        with torch.inference_mode():
            model.embedGameInteraction([bad_interaction], indices, cards)


def test_embed_game_interaction_rejects_out_of_range_deck_id() -> None:
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    game_state = fixtures.EMBED_GAME_STATE_CASES["three_cards"]
    with torch.inference_mode():
        state_emb, _mask, indices = model.embedGameState([game_state])
        cards = fixtures.extract_card_embeddings(state_emb)

    lookup_size = indices.size(1)
    bad_interaction = [_make_target_interaction([lookup_size + 10])]
    with pytest.raises((RuntimeError, ValueError), match="deck_id"):
        with torch.inference_mode():
            model.embedGameInteraction([bad_interaction], indices, cards)


def test_embed_game_interaction_empty_batch_with_nonempty_state() -> None:
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    game_state = fixtures.EMBED_GAME_STATE_CASES["three_cards"]
    with torch.inference_mode():
        state_emb, _mask, indices = model.embedGameState([game_state])
        cards = fixtures.extract_card_embeddings(state_emb)
        out, out_mask = model.embedGameInteraction([[]], indices, cards)
    assert out.shape == (1, 0, DIM)
    assert out_mask.shape == (1, 0)
