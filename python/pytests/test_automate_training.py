"""Tests for rollout policy-iteration automation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PYTHON = _REPO / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

from automate_training import build_iteration_command, runtime_budget_exhausted


def test_iteration_command_contains_rollout_schedule(tmp_path: Path) -> None:
    args = argparse.Namespace(
        rollout_dir=tmp_path / "rollouts",
        model_dir=tmp_path / "models",
        seed_games=100,
        root_states=256,
        train_steps=1000,
        batch_size=128,
        opponent_history=4,
        replay_shards=5,
        collection_mode="auto",
        concurrent_games=32,
        inference_batch_size=16,
        batch_linger_ms=1.0,
        root_action_rollouts=10,
        root_action_temperature=1.0,
        target_contexts_per_action=2,
        target_choices_per_context=3,
        target_rollouts_per_choice=1,
    )

    command = build_iteration_command(args)

    assert command[command.index("--seed-games") + 1] == "100"
    assert command[command.index("--root-states") + 1] == "256"
    assert command[command.index("--train-steps") + 1] == "1000"
    assert command[command.index("--root-action-rollouts") + 1] == "10"
    assert "--resume" not in command
    assert "--start-stage" not in command


def test_runtime_budget_is_checked_before_next_iteration() -> None:
    assert not runtime_budget_exhausted(100.0, None, now=100_000.0)
    assert not runtime_budget_exhausted(100.0, 2.0, now=7_299.9)
    assert runtime_budget_exhausted(100.0, 2.0, now=7_300.0)
