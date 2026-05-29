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
from multi_head_attention import MultiHeadAttentionArgs  # noqa: E402

DIM = 32
DIM_INNER = 16
DIM_INTERACTION_INNER = 16
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 16


def _make_attention_args(
    device: torch.device, dtype: torch.dtype = torch.float32
) -> MultiHeadAttentionArgs:
    return MultiHeadAttentionArgs(
        DIM, DIM, DIM, HEAD_DIM, NUM_HEADS, device=device, dtype=dtype
    )


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def test_kumpel_network_smoke_forward(device: torch.device) -> None:
    game_state_bytes = smoke_fixtures.load_smoke_game_state_bytes()
    interaction_bytes = smoke_fixtures.load_smoke_game_interaction_bytes()

    dtype = torch.float32
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

    with torch.inference_mode():
        scores, _, _, _ = model(game_state_bytes, interaction_bytes)

    assert scores.shape == torch.Size([len(interaction_bytes)])
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
