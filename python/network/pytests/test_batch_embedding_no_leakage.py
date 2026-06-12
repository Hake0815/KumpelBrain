"""Verify batched game-state embedding does not leak cross-game adjacency."""

from __future__ import annotations

import sys
from pathlib import Path

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
from golden_test_utils import deterministic_algorithms, seed_for_device  # noqa: E402

DIM = fixtures.FIXTURE_DIMENSION_OUT
NUM_PLAYER_STATE_ROWS = 2

_ADJACENCY_RELATIONS = (
    "evolves_from_adjacency",
    "attached_energy_adjacency",
    "pre_evolutions_adjacency",
)


def _card_bytes_from_game_state(game_state_bytes: bytes) -> list[bytes]:
    pb2 = fixtures._pb2_mod()
    game_state = pb2.ProtoBufGameState()
    game_state.ParseFromString(game_state_bytes)
    return [card_state.SerializeToString() for card_state in game_state.card_states]


def _dense_adjacency(adjacency, relation_name: str) -> torch.Tensor:
    relation = getattr(adjacency, relation_name)
    return relation.coalesce().cpu().to_dense()


def test_batched_game_state_matches_individual_runs() -> None:
    device = torch.device("cpu")
    payloads = [
        fixtures.EMBED_GAME_STATE_CASES["three_cards"],
        fixtures.EMBED_GAME_STATE_CASES["mixed_relations"],
    ]

    with deterministic_algorithms(True):
        seed_for_device(device)
        model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
        model.eval()
        with torch.inference_mode():
            batched_emb, batched_mask, batched_indices = model.embedGameState(payloads)
            singles = [model.embedGameState([p]) for p in payloads]

    for game_index, (single_emb, single_mask, single_indices) in enumerate(singles):
        single_emb = single_emb[0]
        single_mask = single_mask[0]
        single_indices = single_indices[0]
        batched_row_emb = batched_emb[game_index, : single_emb.size(0)]
        batched_row_mask = batched_mask[game_index, : single_emb.size(0)]
        torch.testing.assert_close(
            batched_row_emb,
            single_emb,
            msg=lambda m: f"game {game_index} embedding mismatch: {m}",
        )
        torch.testing.assert_close(
            batched_row_mask,
            single_mask,
            msg=lambda m: f"game {game_index} mask mismatch: {m}",
        )
        width = max(single_indices.size(0), batched_indices.size(1))
        padded_single = torch.full((width,), -1, dtype=torch.long)
        padded_batched = torch.full((width,), -1, dtype=torch.long)
        padded_single[: single_indices.size(0)] = single_indices.cpu()
        padded_batched[: batched_indices.size(1)] = batched_indices[game_index].cpu()
        torch.testing.assert_close(
            padded_batched,
            padded_single,
            msg=lambda m: f"game {game_index} card_indices mismatch: {m}",
        )


def test_batched_adjacency_is_block_diagonal() -> None:
    device = torch.device("cpu")
    payloads = [
        fixtures.EMBED_GAME_STATE_CASES["three_cards"],
        fixtures.EMBED_GAME_STATE_CASES["mixed_relations"],
    ]
    card_batches = [_card_bytes_from_game_state(payload) for payload in payloads]

    with deterministic_algorithms(True):
        seed_for_device(device)
        card_embedding = kumpel_embedding.make_card_embedding(
            None, DIM, device=device, dtype=torch.float32
        )
        card_embedding.eval()
        with torch.inference_mode():
            _, batched_adjacency, _, _ = card_embedding.forward_batched(card_batches)
            single_adjacencies = [card_embedding.forward(cards)[1] for cards in card_batches]

    segment_sizes = [len(cards) for cards in card_batches]
    total_cards = sum(segment_sizes)
    for relation_name in _ADJACENCY_RELATIONS:
        expected = torch.zeros((total_cards, total_cards), dtype=torch.float32)
        offset = 0
        for segment_size, single_adjacency in zip(segment_sizes, single_adjacencies, strict=True):
            block = _dense_adjacency(single_adjacency, relation_name)
            expected[offset : offset + segment_size, offset : offset + segment_size] = block
            offset += segment_size
        batched = _dense_adjacency(batched_adjacency, relation_name)
        torch.testing.assert_close(
            batched,
            expected,
            msg=lambda m, rel=relation_name: (
                f"batched {rel} must be block-diagonal across games: {m}"
            ),
        )


def test_batched_game_interaction_matches_individual_runs() -> None:
    device = torch.device("cpu")
    case_a = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
    case_b = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_four"]
    game_states = [case_a[0], case_b[0]]
    interactions_per_game = [case_a[1], case_b[1]]

    with deterministic_algorithms(True):
        seed_for_device(device)
        model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
        model.eval()
        with torch.inference_mode():
            batched_state_emb, _, batched_indices = model.embedGameState(game_states)
            batched_cards = batched_state_emb[:, NUM_PLAYER_STATE_ROWS:, :]
            batched_interactions, batched_mask = model.embedGameInteraction(
                interactions_per_game, batched_indices, batched_cards
            )
            singles = []
            for game_state, interactions in zip(game_states, interactions_per_game, strict=True):
                state_emb, _, indices = model.embedGameState([game_state])
                cards = state_emb[:, NUM_PLAYER_STATE_ROWS:, :]
                interaction_emb, interaction_mask = model.embedGameInteraction(
                    [interactions], indices, cards
                )
                singles.append((interaction_emb[0], interaction_mask[0]))

    for game_index, (single_emb, single_mask) in enumerate(singles):
        batched_row_emb = batched_interactions[game_index, : single_emb.size(0)]
        batched_row_mask = batched_mask[game_index, : single_mask.size(0)]
        torch.testing.assert_close(
            batched_row_emb,
            single_emb,
            msg=lambda m: f"game {game_index} interaction embedding mismatch: {m}",
        )
        torch.testing.assert_close(
            batched_row_mask,
            single_mask,
            msg=lambda m: f"game {game_index} interaction mask mismatch: {m}",
        )
