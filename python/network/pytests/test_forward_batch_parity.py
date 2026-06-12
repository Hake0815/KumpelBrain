"""forward_batch contracts and parity checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"
_PYTHON_DIR = _REPO_ROOT / "python"

for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR, _PYTHON_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import kumpel_network  # noqa: E402
import game_embedding_fixtures as fixtures  # noqa: E402
from golden_test_utils import deterministic_algorithms, seed_for_device  # noqa: E402
from multi_head_attention import MultiHeadAttentionArgs  # noqa: E402
from selector import Selector  # noqa: E402

DIM = fixtures.FIXTURE_DIMENSION_OUT
DIM_INNER = DIM * 4
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 8


def _attention_args(device: torch.device) -> MultiHeadAttentionArgs:
    return MultiHeadAttentionArgs(
        DIM, DIM, DIM, HEAD_DIM, NUM_HEADS, device=device, dtype=torch.float32
    )


def _make_network(device: torch.device) -> kumpel_network.KumpelNetwork:
    args = _attention_args(device)
    model = kumpel_network.KumpelNetwork(
        DIM, DIM_INNER, DIM_INNER, args, args, NUM_LAYERS, device=device
    )
    model.eval()
    return model


def _make_selector(device: torch.device) -> Selector:
    selector = Selector(DIM, DIM_INNER, _attention_args(device), device=device)
    selector.eval()
    return selector


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def test_forward_delegates_to_forward_batch(device: torch.device) -> None:
    """forward() slicing matches forward_batch(B=1) valid rows from a single run."""
    model = _make_network(device)
    state, interactions = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]

    with deterministic_algorithms(True):
        seed_for_device(device)
        with torch.inference_mode():
            batch = model.forward_batch([state], [interactions])
            scores, state_out, emb_int, card_idx = model.forward(state, interactions)

    valid_state = batch[5][0]
    valid_int = batch[4][0]
    assert scores.shape == batch[0][0, valid_int].shape
    assert state_out.shape == batch[1][0, valid_state].shape
    assert emb_int.shape == batch[2][0, valid_int].shape
    assert card_idx.shape == batch[3][0].shape
    assert torch.isfinite(scores).all()


def test_forward_batch_duplicate_games_same_transformed_state(device: torch.device) -> None:
    """Identical games in one batch share the same transformed state rows."""
    model = _make_network(device)
    state, interactions = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]

    with deterministic_algorithms(True):
        seed_for_device(device)
        with torch.inference_mode():
            batched = model.forward_batch([state, state], [interactions, interactions])

    valid_state = batched[5][0]
    torch.testing.assert_close(
        batched[1][1, valid_state], batched[1][0, valid_state], rtol=1e-4, atol=1e-4
    )
    assert torch.isfinite(batched[0][batched[4]]).all()


def test_forward_batch_mixed_interaction_counts(device: torch.device) -> None:
    """Batched games with different interaction counts produce valid per-game shapes."""
    model = _make_network(device)
    pair1 = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
    pair2 = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_four"]

    with deterministic_algorithms(True):
        seed_for_device(device)
        with torch.inference_mode():
            batch = model.forward_batch(
                [pair1[0], pair2[0]], [pair1[1], pair2[1]]
            )

    assert batch[0].shape[0] == 2
    assert batch[0].shape[1] == len(pair2[1])
    assert batch[4][0].sum() == len(pair1[1])
    assert batch[4][1].sum() == len(pair2[1])
    assert torch.isfinite(batch[0][batch[4]]).all()


def test_selector_forward_batch_parity(device: torch.device) -> None:
    selector = _make_selector(device)
    torch.manual_seed(1)

    transformed_states = [
        torch.randn(10, DIM, device=device),
        torch.randn(14, DIM, device=device),
    ]
    card_indices = [
        torch.arange(8, device=device, dtype=torch.long),
        torch.arange(12, device=device, dtype=torch.long),
    ]
    embedded_interactions = [
        torch.randn(DIM, device=device),
        torch.randn(DIM, device=device),
    ]
    candidates = [
        torch.tensor([0, 2, 5], device=device, dtype=torch.long),
        torch.tensor([1, 3, 7, 9], device=device, dtype=torch.long),
    ]
    partial = [
        torch.tensor([1], device=device, dtype=torch.long),
        torch.tensor([], device=device, dtype=torch.long),
    ]
    include_stop = [True, False]

    with torch.inference_mode():
        batched = selector.forward_batch(
            candidates,
            partial,
            transformed_states,
            embedded_interactions,
            card_indices,
            include_stop,
        )
        individual = [
            selector(
                candidates[i],
                partial[i],
                transformed_states[i],
                embedded_interactions[i],
                card_indices[i],
                include_stop_token=include_stop[i],
            )
            for i in range(2)
        ]

    for i in range(2):
        n_scores = candidates[i].size(0) + (1 if include_stop[i] else 0)
        torch.testing.assert_close(
            batched[i, :n_scores],
            individual[i],
            rtol=1e-5,
            atol=1e-5,
        )
