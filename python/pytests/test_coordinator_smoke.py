"""Smoke test: BatchedInferenceService matches DirectInferenceClient on move eval."""

from __future__ import annotations

import sys
import threading
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

        direct_scores, direct_state, direct_emb, direct_indices = direct_out
        batched_scores, batched_state, batched_emb, batched_indices = batched_out

        assert direct_scores.value_logits.shape == batched_scores.value_logits.shape
        assert direct_scores.policy_logits.shape == batched_scores.policy_logits.shape
        assert direct_state.shape == batched_state.shape
        assert direct_emb.shape == batched_emb.shape
        assert direct_indices.shape == batched_indices.shape
        assert (
            direct_scores.policy_logits.argmax()
            == batched_scores.policy_logits.argmax()
        )
        assert torch.isfinite(batched_scores.value_logits).all()
        assert torch.isfinite(batched_scores.policy_logits).all()
        torch.testing.assert_close(direct_state, batched_state, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(direct_indices, batched_indices, rtol=0, atol=0)

        service.shutdown()


def test_batched_service_concurrent_two_requests(device: torch.device) -> None:
    with deterministic_algorithms(True):
        seed_for_device(device)
        network, selector, _ = create_self_play_models(device)
        service = BatchedInferenceService(
            network, selector, device, max_batch=4, linger_ms=5
        )
        client = BatchedInferenceClient(service)
        service.register_game()
        service.register_game()

        state, interactions = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
        assert client.tensor_device == device
        barrier = threading.Barrier(2)
        results: list = []
        errors: list[BaseException] = []

        def _submit() -> None:
            try:
                barrier.wait(timeout=5.0)
                with torch.inference_mode():
                    results.append(client.evaluate_move(state, interactions))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)

        assert not errors
        assert len(results) == 2
        assert (
            results[0][0].policy_logits.argmax()
            == results[1][0].policy_logits.argmax()
        )
        torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-5, atol=1e-5)
        stats = service.batch_stats()
        assert stats.move_requests == 2
        assert stats.move_batches == 1
        assert stats.move_max_batch == 2

        service.shutdown()


def test_batched_service_target_score_matches_direct(device: torch.device) -> None:
    with deterministic_algorithms(True):
        seed_for_device(device)
        network, selector, _ = create_self_play_models(device)
        direct = DirectInferenceClient(network, selector, device)
        service = BatchedInferenceService(
            network, selector, device, max_batch=4, linger_ms=1
        )
        client = BatchedInferenceClient(service)
        service.register_game()

        state, interactions = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
        with torch.inference_mode():
            _, transformed_state, embedded_interactions, card_indices = direct.evaluate_move(
                state, interactions
            )

        dim = transformed_state.size(-1)
        num_cards = int(card_indices.ge(0).sum().item())
        candidates = torch.arange(min(3, num_cards), dtype=torch.long)
        partial = torch.tensor([], dtype=torch.long)

        with torch.inference_mode():
            direct_scores = direct.score_targets(
                candidates,
                partial,
                transformed_state,
                embedded_interactions[0],
                card_indices,
                include_stop_token=False,
            )
            batched_scores = client.score_targets(
                candidates,
                partial,
                transformed_state,
                embedded_interactions[0],
                card_indices,
                include_stop_token=False,
            )

        assert (
            batched_scores.value_logits.shape == direct_scores.value_logits.shape
        )
        assert batched_scores.value_logits.device == device
        torch.testing.assert_close(
            direct_scores.value_logits,
            batched_scores.value_logits,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            direct_scores.policy_logits,
            batched_scores.policy_logits,
            rtol=1e-5,
            atol=1e-5,
        )
        service.shutdown()


def test_batched_service_rejects_submission_after_shutdown(device: torch.device) -> None:
    with deterministic_algorithms(True):
        seed_for_device(device)
        network, selector, _ = create_self_play_models(device)
        service = BatchedInferenceService(
            network, selector, device, max_batch=4, linger_ms=1
        )
        client = BatchedInferenceClient(service)
        state, interactions = fixtures.EMBED_GAME_INTERACTION_CASES["all_types_one"]
        service.shutdown()

        with pytest.raises(RuntimeError, match="shut down"):
            client.evaluate_move(state, interactions)
