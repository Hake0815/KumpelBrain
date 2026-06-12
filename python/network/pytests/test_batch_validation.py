"""Validation and API-contract tests for batched C++ embeddings."""

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


def test_embed_game_state_accepts_raw_bytes() -> None:
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    payload = fixtures.EMBED_GAME_STATE_CASES["three_cards"]
    with torch.inference_mode():
        list_out, list_mask, list_indices = model.embedGameState([payload])
        bytes_out, bytes_mask, bytes_indices = model.embedGameState(payload)
    torch.testing.assert_close(list_out[0], bytes_out[0])
    torch.testing.assert_close(list_mask[0], bytes_mask[0])
    torch.testing.assert_close(list_indices[0], bytes_indices[0])


def test_embed_game_interaction_rejects_mismatched_batch_shapes() -> None:
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    state_a = fixtures.EMBED_GAME_STATE_CASES["three_cards"]
    state_b = fixtures.EMBED_GAME_STATE_CASES["mixed_relations"]
    interaction = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"][1]
    with torch.inference_mode():
        emb, _mask, indices = model.embedGameState([state_a, state_b])
        cards = fixtures.extract_card_embeddings(emb)
        wrong_cards = cards[:1]
    with pytest.raises((RuntimeError, ValueError), match="batch"):
        with torch.inference_mode():
            model.embedGameInteraction([interaction, interaction], indices, wrong_cards)


def test_embed_game_interaction_nested_tuple_batch() -> None:
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    state, interactions = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
    with torch.inference_mode():
        emb, _mask, indices = model.embedGameState([state])
        cards = fixtures.extract_card_embeddings(emb)
        list_out, list_mask = model.embedGameInteraction([interactions], indices, cards)
        tuple_out, tuple_mask = model.embedGameInteraction((interactions,), indices, cards)
    torch.testing.assert_close(list_out, tuple_out)
    torch.testing.assert_close(list_mask, tuple_mask)


def test_batched_interaction_rejects_cross_game_deck_id() -> None:
    pb2 = proto_serialization._load_proto_module()
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()

    state_a = fixtures.EMBED_GAME_STATE_CASES["three_cards"]
    state_b = fixtures.EMBED_GAME_STATE_CASES["mixed_relations"]

    interaction = pb2.ProtoBufGameInteraction()
    interaction.type = pb2.GAME_INTERACTION_TYPE_SELECT_CARDS
    data = interaction.data.add()
    data.data_type = pb2.GAME_INTERACTION_DATA_TYPE_TARGET_DATA
    target = data.target_data
    target.possible_targets.extend([99])
    target.target_action = pb2.ACTION_ON_SELECTION_DISCARD
    target.remainder_action = pb2.ACTION_ON_SELECTION_TAKE_TO_HAND
    target.number_of_targets = 1
    bad_interaction = [interaction.SerializeToString()]

    with torch.inference_mode():
        emb, _mask, indices = model.embedGameState([state_a, state_b])
        cards = fixtures.extract_card_embeddings(emb)

    with pytest.raises((RuntimeError, ValueError), match="deck_id"):
        with torch.inference_mode():
            model.embedGameInteraction([bad_interaction, []], indices, cards)
