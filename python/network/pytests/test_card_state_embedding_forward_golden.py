"""Regression goldens for C++ CardStateEmbedding.forward (CPU and CUDA).

Regenerate after intentional math changes:

  .venv/bin/python python/network/pytests/generate_card_state_embedding_forward_goldens.py

Run: python -m pytest python/network/pytests/test_card_state_embedding_forward_golden.py -v
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"
_FIXTURES_DIR = _PYTESTS_DIR / "fixtures"

for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import pytest
import torch

import card_state_embedding_forward_fixtures as fixtures
import kumpel_embedding  # noqa: E402

GOLDEN_SEED = 42
RTOL = 0.0
ATOL = 1e-5


@contextlib.contextmanager
def _deterministic_algorithms(enabled: bool):
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


def _seed(device: torch.device) -> None:
    torch.manual_seed(GOLDEN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(GOLDEN_SEED)


def _load_golden(name: str) -> dict[str, torch.Tensor]:
    path = _FIXTURES_DIR / name
    if not path.is_file():
        pytest.skip(f"missing golden file: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda", 0)
    return torch.device("cpu")


@pytest.fixture
def golden(device: torch.device) -> dict[str, torch.Tensor]:
    fname = (
        "card_state_embedding_forward_golden_cuda.pt"
        if device.type == "cuda"
        else "card_state_embedding_forward_golden_cpu.pt"
    )
    return _load_golden(fname)


@pytest.mark.parametrize("case_id", sorted(fixtures.FIXTURE_CASES.keys()))
def test_card_state_embedding_forward_golden_case(
    case_id: str,
    device: torch.device,
    golden: dict[str, torch.Tensor],
):
    expected = golden[case_id]
    states = fixtures.FIXTURE_CASES[case_id]

    with _deterministic_algorithms(True):
        _seed(device)
        model = kumpel_embedding.CardStateEmbedding(
            fixtures.FIXTURE_DIMENSION_OUT, device=device, dtype=torch.float32
        )
        model.eval()
        with torch.inference_mode():
            actual = model.forward(states)
        if device.type == "cuda":
            torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=RTOL,
        atol=ATOL,
        msg=lambda msg: f"{case_id} on {device}: {msg}",
    )


def test_pre_evolutions_adjacency_differs_from_attached_energy_adjacency():
    """Regression: pre_evolutions must not be built from attached_energy_cards_matrix."""
    device = torch.device("cpu")
    with _deterministic_algorithms(True):
        _seed(device)
        shared = kumpel_embedding.SharedEmbeddingHolder(
            fixtures.FIXTURE_DIMENSION_OUT, device=device, dtype=torch.float32
        )
        emb = kumpel_embedding.CardEmbedding(
            shared,
            fixtures.FIXTURE_DIMENSION_OUT,
            device=device,
            dtype=torch.float32,
        )
        emb.eval()
        card_bytes = fixtures.build_adjacency_divergent_card_bytes()
        with torch.inference_mode():
            _h, adj = emb.forward(card_bytes)

    pre = adj.pre_evolutions_adjacency.coalesce().cpu().to_dense()
    att = adj.attached_energy_adjacency.coalesce().cpu().to_dense()
    assert not torch.allclose(pre, att, rtol=0.0, atol=0.0), (
        "pre_evolutions_adjacency must differ from attached_energy_adjacency on divergent batch"
    )
