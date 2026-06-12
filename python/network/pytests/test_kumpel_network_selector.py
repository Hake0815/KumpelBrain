"""Unit tests for KumpelNetwork and Selector tensor contracts."""

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

import kumpel_network  # noqa: E402
import kumpel_network_json_fixtures as smoke_fixtures  # noqa: E402
from selector import Selector  # noqa: E402
from game_embedding import NUM_PLAYER_STATE_ROWS  # noqa: E402
from multi_head_attention import MultiHeadAttentionArgs  # noqa: E402

DIM = 32
DIM_INNER = 16
DIM_INTERACTION_INNER = 16
DIM_TARGET_INNER = 16
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 16
NUM_CARDS = 6


def _make_attention_args(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> MultiHeadAttentionArgs:
    return MultiHeadAttentionArgs(
        DIM, DIM, DIM, HEAD_DIM, NUM_HEADS, device=device, dtype=dtype
    )


def _make_model(device: torch.device, dtype: torch.dtype = torch.float32):
    attention_args = _make_attention_args(device, dtype)
    model = kumpel_network.KumpelNetwork(
        DIM,
        DIM_INNER,
        DIM_INTERACTION_INNER,
        attention_args,
        attention_args,
        NUM_LAYERS,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def _make_selector(device: torch.device, dtype: torch.dtype = torch.float32):
    attention_args = _make_attention_args(device, dtype)
    selector = Selector(
        DIM,
        DIM_TARGET_INNER,
        attention_args,
        device=device,
        dtype=dtype,
    )
    selector.eval()
    return selector


def _synthetic_tensors(device: torch.device, dtype: torch.dtype):
    state_len = NUM_PLAYER_STATE_ROWS + NUM_CARDS
    transformed_state = torch.randn(state_len, DIM, device=device, dtype=dtype)
    card_indices = torch.arange(NUM_CARDS, device=device, dtype=torch.long)
    embedded_interaction = torch.randn(DIM, device=device, dtype=dtype)
    return transformed_state, card_indices, embedded_interaction


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_multi_head_attention_args_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError, match="d_q must be positive"):
        MultiHeadAttentionArgs(0, 8, 8, 4, 2)
    with pytest.raises(ValueError, match="nheads must be positive"):
        MultiHeadAttentionArgs(8, 8, 8, 4, 0)


def test_forward_returns_2d_transformed_state(device: torch.device) -> None:
    game_state_bytes = smoke_fixtures.load_smoke_game_state_bytes()
    interaction_bytes = smoke_fixtures.load_smoke_game_interaction_bytes()
    model = _make_model(device)

    with torch.inference_mode():
        scores, transformed_state, embedded_interactions, card_indices = model(
            game_state_bytes, interaction_bytes
        )

    assert scores.shape == torch.Size([len(interaction_bytes)])
    assert transformed_state.dim() == 2
    assert embedded_interactions.dim() == 2
    assert card_indices.dim() == 1


def test_selector_accepts_forward_output(device: torch.device) -> None:
    game_state_bytes = smoke_fixtures.load_smoke_game_state_bytes()
    interaction_bytes = smoke_fixtures.load_smoke_game_interaction_bytes()
    model = _make_model(device)
    selector = _make_selector(device)

    with torch.inference_mode():
        _, transformed_state, embedded_interactions, card_indices = model(
            game_state_bytes, interaction_bytes
        )
        first_interaction = embedded_interactions[0]
        candidates = card_indices[:3]
        partial_selection = torch.tensor([], device=device, dtype=torch.long)
        target_scores = selector(
            candidates,
            partial_selection,
            transformed_state,
            first_interaction,
            card_indices,
            include_stop_token=False,
        )

    assert target_scores.shape == torch.Size([candidates.numel()])
    assert torch.isfinite(target_scores).all()


@pytest.mark.parametrize("include_stop_token", [False, True])
def test_selector_synthetic_shapes(
    device: torch.device, include_stop_token: bool
) -> None:
    selector = _make_selector(device)
    transformed_state, card_indices, embedded_interaction = _synthetic_tensors(
        device, torch.float32
    )
    candidates = torch.tensor([0, 2, 4], device=device, dtype=torch.long)
    partial_selection = torch.tensor([1], device=device, dtype=torch.long)

    with torch.inference_mode():
        scores = selector(
            candidates,
            partial_selection,
            transformed_state,
            embedded_interaction,
            card_indices,
            include_stop_token=include_stop_token,
        )

    assert scores.shape == torch.Size([candidates.numel()])
    assert torch.isfinite(scores).all()


def test_selector_empty_partial_selection(device: torch.device) -> None:
    selector = _make_selector(device)
    transformed_state, card_indices, embedded_interaction = _synthetic_tensors(
        device, torch.float32
    )
    candidates = torch.tensor([0, 1], device=device, dtype=torch.long)
    partial_selection = torch.tensor([], device=device, dtype=torch.long)

    with torch.inference_mode():
        scores = selector(
            candidates,
            partial_selection,
            transformed_state,
            embedded_interaction,
            card_indices,
            include_stop_token=True,
        )

    assert scores.shape == torch.Size([2])
