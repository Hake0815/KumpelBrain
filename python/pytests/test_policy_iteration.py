"""Champion gate and phase-balanced sampling tests."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for path in (
    _REPO / "cpp" / "build",
    _REPO / "python",
    _REPO / "python" / "network",
    _REPO / "python" / "game_logic_wrappers",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from training.policy_iteration import (
    OpponentPool,
    run_champion_gate,
    sample_phase_balanced_roots,
)
from training.rollout_data import RootState


def _root(index: int, phase: str) -> RootState:
    return RootState(
        root_id=f"{index:064x}",
        state_bytes=b"s",
        deck_list1={},
        deck_list2={},
        player1_name="p1",
        player2_name="p2",
        player_name="p1",
        interaction_bytes=[b"a"],
        ply_from_end=index,
        phase=phase,
    )


def test_phase_balanced_sampling_round_robins_available_phases() -> None:
    roots = [
        _root(0, "terminal"),
        _root(1, "terminal"),
        _root(3, "late"),
        _root(8, "mid_late"),
        _root(16, "mid"),
        _root(32, "early"),
    ]

    selected = sample_phase_balanced_roots(roots, 5, seed=0)

    assert {root.phase for root in selected} == {
        "terminal",
        "late",
        "mid_late",
        "mid",
        "early",
    }


def test_gate_promotes_strong_candidate_after_minimum_games() -> None:
    def play_block(count: int, _candidate_is_player1: bool) -> list[float]:
        return [1.0] * count

    result = run_champion_gate(play_block)

    assert result.promoted
    assert result.games == 200
    assert result.interval.lower > 0.5


def test_gate_rejects_weak_candidate_after_minimum_games() -> None:
    def play_block(count: int, _candidate_is_player1: bool) -> list[float]:
        return [0.0] * count

    result = run_champion_gate(play_block)

    assert not result.promoted
    assert result.games == 200
    assert result.interval.upper <= 0.5


def test_gate_rejects_inconclusive_candidate_at_maximum() -> None:
    flip = False

    def play_block(count: int, _candidate_is_player1: bool) -> list[float]:
        nonlocal flip
        values = []
        for _ in range(count):
            values.append(1.0 if flip else 0.0)
            flip = not flip
        return values

    result = run_champion_gate(play_block)

    assert not result.promoted
    assert result.games == 800


def test_opponent_pool_uses_champion_then_history_branch() -> None:
    class FakeRandom:
        def __init__(self):
            self.values = iter([0.74, 0.75])

        def random(self) -> float:
            return next(self.values)

        def choice(self, values):
            return values[-1]

    pool = object.__new__(OpponentPool)
    pool.champion = "champion"
    pool.history = ["old-1", "old-2"]
    pool.rng = FakeRandom()

    assert pool.sample() == "champion"
    assert pool.sample() == "old-2"
