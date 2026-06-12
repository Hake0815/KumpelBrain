"""CUDA embedding device: coordinator wiring and GPU forward smoke tests."""

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

import game_embedding_fixtures as fixtures  # noqa: E402
import kumpel_network  # noqa: E402
from golden_test_utils import deterministic_algorithms, seed_for_device  # noqa: E402
from inference_models import create_self_play_models  # noqa: E402
from multi_head_attention import MultiHeadAttentionArgs  # noqa: E402

DIM = fixtures.FIXTURE_DIMENSION_OUT
DIM_INNER = DIM * 4
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 8


def _attention_args(device: torch.device) -> MultiHeadAttentionArgs:
    return MultiHeadAttentionArgs(
        DIM, DIM, DIM, HEAD_DIM, NUM_HEADS, device=device, dtype=torch.float32
    )


def _make_network(
    device: torch.device, embedding_device: torch.device
) -> kumpel_network.KumpelNetwork:
    args = _attention_args(device)
    model = kumpel_network.KumpelNetwork(
        DIM,
        DIM_INNER,
        DIM_INNER,
        args,
        args,
        NUM_LAYERS,
        device=device,
        embedding_device=embedding_device,
    )
    model.eval()
    return model


def test_default_embedding_device_is_cpu() -> None:
    network, _, device = create_self_play_models(torch.device("cpu"))
    assert network.embedding_device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_embed_on_compute_device_uses_cuda_embedding() -> None:
    network, _, device = create_self_play_models(
        torch.device("cuda"), embed_on_compute_device=True
    )
    assert device.type == "cuda"
    assert network.embedding_device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_embedding_device_cuda_forward_batch_runs() -> None:
    device = torch.device("cuda", 0)
    pair1 = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
    pair2 = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_four"]
    game_states = [pair1[0], pair2[0]]
    interactions = [pair1[1], pair2[1]]

    with deterministic_algorithms(True):
        seed_for_device(device)
        model = _make_network(device, device)

        with torch.inference_mode():
            scores, state, emb_int, card_idx, int_mask, state_mask = (
                model.forward_batch(game_states, interactions)
            )

    assert scores.shape[0] == 2
    assert scores.shape[1] == len(pair2[1])
    assert int_mask[0].sum() == len(pair1[1])
    assert int_mask[1].sum() == len(pair2[1])
    assert torch.isfinite(scores[int_mask]).all()
    assert torch.isfinite(state[state_mask]).all()
    assert torch.isfinite(emb_int[int_mask]).all()
    assert card_idx.device.type == "cuda"
