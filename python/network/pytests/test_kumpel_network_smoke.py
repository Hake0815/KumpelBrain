"""Smoke test for KumpelNetwork on real ProtoBuf game state / interaction JSON."""

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

DIM = 32
DIM_INNER = 16
DIM_INTERACTION_INNER = 16
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 16


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_kumpel_network_smoke_forward(device: torch.device) -> None:
    game_state_bytes = smoke_fixtures.load_smoke_game_state_bytes()
    interaction_bytes = smoke_fixtures.load_smoke_game_interaction_bytes()

    model = kumpel_network.KumpelNetwork(
        DIM,
        DIM_INNER,
        DIM_INTERACTION_INNER,
        HEAD_DIM,
        NUM_HEADS,
        NUM_LAYERS,
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    with torch.inference_mode():
        scores = model(game_state_bytes, interaction_bytes)

    print(scores)
    assert scores.shape == torch.Size([len(interaction_bytes)])
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()
