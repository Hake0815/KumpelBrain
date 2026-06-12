"""Smoke test: BatchedInferenceService matches DirectInferenceClient on move eval."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
_NETWORK_DIR = _PYTHON_DIR / "network"
_PYTESTS_DIR = _NETWORK_DIR / "pytests"
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"

for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_DIR, _PYTHON_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import game_embedding_fixtures as fixtures  # noqa: E402
from golden_test_utils import deterministic_algorithms, seed_for_device  # noqa: E402
from inference_models import create_self_play_models  # noqa: E402
from inference_service import (  # noqa: E402
    BatchedInferenceClient,
    BatchedInferenceService,
    DirectInferenceClient,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def test_batched_service_matches_direct_client(device: torch.device) -> None:
    with deterministic_algorithms(True):
        seed_for_device(device)
        network, selector, _ = create_self_play_models(device)
        direct = DirectInferenceClient(network, selector, device)
        service = BatchedInferenceService(
            network, selector, device, max_batch=4, linger_ms=1
        )
        batched = BatchedInferenceClient(service)
        service.register_game()
        service.register_game()

        state, interactions = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]

        with torch.inference_mode():
            direct_out = direct.evaluate_move(state, interactions)
            batched_out = batched.evaluate_move(state, interactions)

        for d_tensor, b_tensor in zip(direct_out, batched_out):
            assert d_tensor.shape == b_tensor.shape
            assert torch.isfinite(d_tensor).all()
            assert torch.isfinite(b_tensor).all()
        assert direct_out[0].argmax() == batched_out[0].argmax()

        service.shutdown()


def test_batched_service_batches_two_requests(device: torch.device) -> None:
    with deterministic_algorithms(True):
        seed_for_device(device)
        network, selector, _ = create_self_play_models(device)
        service = BatchedInferenceService(
            network, selector, device, max_batch=4, linger_ms=5
        )
        client = BatchedInferenceClient(service)
        service.register_game()
        service.register_game()

        pair = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
        state, interactions = pair

        with torch.inference_mode():
            results = [
                client.evaluate_move(state, interactions),
                client.evaluate_move(state, interactions),
            ]

        for scores, _, _, _ in results:
            assert scores.ndim == 1
            assert scores.numel() == len(interactions)
            assert torch.isfinite(scores).all()

        service.shutdown()


def test_batched_service_target_score_cpu_indices(device: torch.device) -> None:
    """Game threads submit CPU index tensors; service runs selector on compute device."""
    with deterministic_algorithms(True):
        seed_for_device(device)
        network, selector, _ = create_self_play_models(device)
        service = BatchedInferenceService(
            network, selector, device, max_batch=4, linger_ms=1
        )
        client = BatchedInferenceClient(service)
        service.register_game()

        dim = 128
        transformed_state = torch.randn(10, dim)
        embedded_interaction = torch.randn(dim)
        card_indices = torch.arange(8, dtype=torch.long)
        candidates = torch.tensor([0, 2, 5], dtype=torch.long)
        partial = torch.tensor([1], dtype=torch.long)

        with torch.inference_mode():
            scores = client.score_targets(
                candidates,
                partial,
                transformed_state,
                embedded_interaction,
                card_indices,
                include_stop_token=False,
            )

        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

        service.shutdown()
