"""Regression goldens for C++ CardEmbedding.forward (CPU and CUDA).

Requires built kumpel_embedding (cpp/build) and fixture files under python/network/fixtures/.

Regenerate goldens after intentional math changes:

  .venv/bin/python python/network/generate_card_embedding_forward_goldens.py

If CUDA tests fail with nondeterministic-algorithm errors, try:

  export CUBLAS_WORKSPACE_CONFIG=:4096:8

Run: python -m pytest python/network/test_card_embedding_forward_golden.py -v
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_NETWORK_DIR = Path(__file__).resolve().parent
_CPP_BUILD = Path(__file__).resolve().parents[2] / "cpp" / "build"
_FIXTURES_DIR = _NETWORK_DIR / "fixtures"

sys.path.insert(0, str(_NETWORK_DIR))
sys.path.insert(0, str(_CPP_BUILD))

import pytest
import torch

import card_embedding_forward_fixtures as fixtures
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
        "card_embedding_forward_golden_cuda.pt"
        if device.type == "cuda"
        else "card_embedding_forward_golden_cpu.pt"
    )
    return _load_golden(fname)


@pytest.mark.parametrize("case_id", sorted(fixtures.FIXTURE_CASES.keys()))
def test_card_embedding_forward_golden_case(
    case_id: str,
    device: torch.device,
    golden: dict[str, torch.Tensor],
):
    expected = golden[case_id]
    cards = fixtures.FIXTURE_CASES[case_id]

    with _deterministic_algorithms(True):
        _seed(device)
        model = kumpel_embedding.CardEmbedding(
            fixtures.FIXTURE_DIMENSION_OUT, device=device, dtype=torch.float32
        )
        model.eval()
        with torch.inference_mode():
            actual, _adjacency = model.forward(cards)
        if device.type == "cuda":
            torch.cuda.synchronize()

    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=RTOL,
        atol=ATOL,
        msg=lambda msg: f"{case_id} on {device}: {msg}",
    )
