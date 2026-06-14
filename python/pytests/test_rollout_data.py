"""Rollout records, branching, targets, and stable split tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
for path in (_REPO / "python", _REPO / "python" / "network"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.rollout_data import (
    ActionEstimate,
    OutcomeCounts,
    RolloutShard,
    RolloutStore,
    RootState,
    improved_policy,
    is_validation_root,
    phase_budget,
    select_branches,
    stable_root_id,
)


def _root() -> RootState:
    return RootState(
        root_id=stable_root_id(b"state", "player1"),
        state_bytes=b"state",
        deck_list1={"a": 1},
        deck_list2={"b": 1},
        player1_name="player1",
        player2_name="player2",
        player_name="player1",
        interaction_bytes=[b"a", b"b"],
        ply_from_end=9,
        phase="mid_late",
        action_estimates=[
            ActionEstimate(0, OutcomeCounts(wins=3, losses=1)),
            ActionEstimate(1, OutcomeCounts(wins=1, losses=3)),
        ],
    )


def test_beta_smoothed_target_and_confidence() -> None:
    outcomes = OutcomeCounts(wins=2, draws=1, losses=1)

    assert outcomes.value_target == pytest.approx(3.5 / 6.0)
    assert outcomes.confidence_weight == pytest.approx(4.0 / 6.0)


def test_improved_policy_prefers_higher_rollout_value() -> None:
    policy = improved_policy(_root().action_estimates)

    assert policy.sum().item() == pytest.approx(1.0)
    assert policy[0] > policy[1]


def test_top_plus_random_branching_is_unique_and_seeded() -> None:
    logits = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    budget = phase_budget(9)

    first = select_branches(logits, budget, rng=random.Random(3))
    second = select_branches(logits, budget, rng=random.Random(3))

    assert first == second
    assert first[:4] == [0, 1, 2, 3]
    assert len(first) == len(set(first)) == 6


def test_rollout_shard_round_trip_stores_root_once(tmp_path: Path) -> None:
    store = RolloutStore(tmp_path)
    shard = RolloutShard(1, "champion", [_root()], {"games": 3})

    loaded = store.load(store.save(shard))

    assert loaded.iteration == 1
    assert len(loaded.roots) == 1
    assert loaded.roots[0].state_bytes == b"state"
    assert loaded.roots[0].action_estimates[0].outcomes.wins == 3


def test_stable_root_split_is_content_based() -> None:
    root_id = stable_root_id(b"same", "player1")

    assert is_validation_root(root_id) == is_validation_root(root_id)
    assert root_id != stable_root_id(b"different", "player1")
