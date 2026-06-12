"""Regression goldens for C++ GameEmbedding.embedGameInteraction (CPU and CUDA).

Regenerate after intentional math changes:

  .venv/bin/python python/network/pytests/generate_game_embedding_goldens.py

Run: python -m pytest python/network/pytests/test_game_embedding_embed_game_interaction_golden.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"

for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import pytest
import torch

import game_embedding_fixtures as fixtures
import kumpel_embedding  # noqa: E402
from golden_test_utils import (  # noqa: E402
    ATOL,
    STRICT_RTOL,
    deterministic_algorithms,
    load_golden,
    seed_for_device,
)


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
        "game_embedding_embed_game_interaction_golden_cuda.pt"
        if device.type == "cuda"
        else "game_embedding_embed_game_interaction_golden_cpu.pt"
    )
    try:
        return load_golden(fname)
    except FileNotFoundError:
        pytest.skip(f"missing golden file: {fname}")


@pytest.mark.parametrize("case_id", sorted(fixtures.EMBED_GAME_INTERACTION_CASES.keys()))
def test_game_embedding_embed_game_interaction_golden_case(
    case_id: str,
    device: torch.device,
    golden: dict[str, torch.Tensor],
):
    expected = golden[case_id]
    game_state_bytes, interaction_bytes = fixtures.EMBED_GAME_INTERACTION_CASES[case_id]

    with deterministic_algorithms(True):
        seed_for_device(device)
        model = kumpel_embedding.GameEmbedding(
            fixtures.FIXTURE_DIMENSION_OUT, device=device, dtype=torch.float32
        )
        model.eval()
        with torch.inference_mode():
            game_state_embedding, _mask, indices = model.embedGameState([game_state_bytes])
            cards = fixtures.extract_card_embeddings(game_state_embedding)
            actual, _interaction_mask = model.embedGameInteraction(
                [interaction_bytes], indices, cards
            )
            actual = actual[0]
        if device.type == "cuda":
            torch.cuda.synchronize()

    expected_indices = fixtures.build_expected_game_state_card_indices(
        game_state_bytes, device
    )
    assert indices.dtype == torch.long
    assert indices.device.type == device.type
    assert indices.shape == expected_indices.shape
    torch.testing.assert_close(
        indices.cpu(),
        expected_indices.cpu(),
        msg=lambda msg: f"{case_id} card_indices on {device}: {msg}",
    )
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=STRICT_RTOL,
        atol=ATOL,
        msg=lambda msg: f"{case_id} on {device}: {msg}",
    )
