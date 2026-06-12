"""Padding-mask parity: batched B=2 with masks equals two independent B=1 runs."""

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

from interaction_network import InteractionNetwork  # noqa: E402
from multi_head_attention import MultiHeadAttentionArgs  # noqa: E402
from state_transformer import StateTransformer  # noqa: E402

DIM = 32
DIM_INNER = 64
NUM_HEADS = 4
HEAD_DIM = 8
NUM_LAYERS = 2


def _attention_args(device: torch.device) -> MultiHeadAttentionArgs:
    return MultiHeadAttentionArgs(
        DIM, DIM, DIM, HEAD_DIM, NUM_HEADS, device=device, dtype=torch.float32
    )


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def test_state_transformer_padding_mask_parity(device: torch.device) -> None:
    torch.manual_seed(42)
    transformer = StateTransformer(
        DIM, DIM_INNER, _attention_args(device), NUM_LAYERS, device=device
    )
    transformer.eval()

    x1 = torch.randn(1, 8, DIM, device=device)
    x2 = torch.randn(1, 12, DIM, device=device)

    with torch.inference_mode():
        out1 = transformer(x1)
        out2 = transformer(x2)

        max_len = 12
        padded = torch.zeros(2, max_len, DIM, device=device)
        padded[0, :8] = x1[0]
        padded[1, :12] = x2[0]
        mask = torch.zeros(2, max_len, device=device, dtype=torch.bool)
        mask[0, :8] = True
        mask[1, :12] = True
        batched = transformer(padded, key_padding_mask=mask)

    torch.testing.assert_close(batched[0, :8], out1[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batched[1, :12], out2[0], rtol=1e-5, atol=1e-5)


def test_interaction_network_padding_mask_parity(device: torch.device) -> None:
    torch.manual_seed(7)
    network = InteractionNetwork(
        DIM, DIM_INNER, _attention_args(device), device=device
    )
    network.eval()

    queries1 = torch.randn(1, 3, DIM, device=device)
    state1 = torch.randn(1, 10, DIM, device=device)
    queries2 = torch.randn(1, 5, DIM, device=device)
    state2 = torch.randn(1, 14, DIM, device=device)

    with torch.inference_mode():
        out1 = network(queries1, state1)
        out2 = network(queries2, state2)

        max_q = 5
        max_s = 14
        queries = torch.zeros(2, max_q, DIM, device=device)
        state = torch.zeros(2, max_s, DIM, device=device)
        queries[0, :3] = queries1[0]
        queries[1, :5] = queries2[0]
        state[0, :10] = state1[0]
        state[1, :14] = state2[0]
        key_mask = torch.zeros(2, max_s, device=device, dtype=torch.bool)
        key_mask[0, :10] = True
        key_mask[1, :14] = True
        batched = network(queries, state, key_mask=key_mask)

    torch.testing.assert_close(batched[0, :3], out1[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batched[1, :5], out2[0], rtol=1e-5, atol=1e-5)


def test_all_true_mask_matches_no_mask(device: torch.device) -> None:
    torch.manual_seed(0)
    transformer = StateTransformer(
        DIM, DIM_INNER, _attention_args(device), NUM_LAYERS, device=device
    )
    transformer.eval()
    x = torch.randn(1, 6, DIM, device=device)
    all_true = torch.ones(1, 6, device=device, dtype=torch.bool)

    with torch.inference_mode():
        without_mask = transformer(x)
        with_mask = transformer(x, key_padding_mask=all_true)

    torch.testing.assert_close(without_mask, with_mask, rtol=0, atol=0)
