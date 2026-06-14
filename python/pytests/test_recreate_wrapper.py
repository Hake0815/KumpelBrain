"""Opt-in C# integration test for forced rollout recreation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from csharp_test_utils import csharp_integration_available

_REPO = Path(__file__).resolve().parents[2]
for path in (
    _REPO / "cpp" / "build",
    _REPO / "python",
    _REPO / "python" / "network",
    _REPO / "python" / "network" / "pytests",
    _REPO / "python" / "training",
    _REPO / "python" / "game_logic_wrappers",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytestmark = pytest.mark.skipif(
    not csharp_integration_available(),
    reason="C# integration disabled; set KUMPEL_ENABLE_CSHARP_TESTS=1 to run",
)


def test_seed_root_can_force_an_interaction_and_continue() -> None:
    from inference_models import create_self_play_models
    from inference_service import DirectInferenceClient
    from main import create_deck_list
    from training.gameplay import run_seed_game
    from training.rollout_runner import ContinuationRolloutRunner

    network, selector, device = create_self_play_models(torch.device("cpu"))
    inference = DirectInferenceClient(network, selector, device)
    roots = run_seed_game(
        deck_list1=create_deck_list(),
        deck_list2=create_deck_list(),
        player1_name="player1",
        player2_name="player2",
        champion=inference,
        opponent=inference,
        compute_device=device,
        seed=0,
    )
    if not roots:
        pytest.skip("Seed game produced no recreatable trainable roots")

    root = roots[len(roots) // 2]
    result = ContinuationRolloutRunner(
        inference,
        lambda: inference,
        device,
        seed=1,
    ).run(root, 0)

    assert result.success, result.error
    assert result.score in {0.0, 0.5, 1.0}
