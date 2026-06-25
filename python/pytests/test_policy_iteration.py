"""Champion gate, phase-balanced sampling, and sampled rollout tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
for path in (
    _REPO / "cpp" / "build",
    _REPO / "python",
    _REPO / "python" / "network",
    _REPO / "python" / "game_logic_wrappers",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from network.action_scores import ActionScores
from training.policy_iteration import (
    OpponentPool,
    RolloutEvaluator,
    run_champion_gate,
    sample_phase_balanced_roots,
)
from training.rollout_data import RootState
from training.rollout_runner import ForcedRolloutResult, ObservedTargetContext


def _root(index: int, phase: str, *, interactions: int = 1) -> RootState:
    return RootState(
        root_id=f"{index:064x}",
        state_bytes=b"s",
        deck_list1={},
        deck_list2={},
        player1_name="p1",
        player2_name="p2",
        player_name="p1",
        interaction_bytes=[bytes(f"{value}", "ascii") for value in range(interactions)],
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


class FakeInference:
    def __init__(self, policy_logits: list[float]):
        self.policy_logits = torch.tensor(policy_logits, dtype=torch.float32)
        self.target_logits = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)

    def evaluate_move(self, state_bytes: bytes, interactions_bytes: list[bytes]):
        interaction_count = len(interactions_bytes)
        del state_bytes, interactions_bytes
        return (
            ActionScores(self.policy_logits, self.policy_logits),
            torch.zeros(4),
            torch.zeros(interaction_count, 4),
            torch.zeros(8, dtype=torch.long),
        )

    def score_targets(
        self,
        candidates,
        partial_selection,
        transformed_state,
        embedded_interaction,
        card_indices,
        include_stop_token,
    ):
        target_count = int(candidates.numel()) + int(include_stop_token)
        del (
            partial_selection,
            transformed_state,
            embedded_interaction,
            card_indices,
            include_stop_token,
        )
        target_logits = torch.arange(float(target_count), 0.0, -1.0)
        return ActionScores(target_logits, target_logits)


class FakeRunner:
    def __init__(
        self,
        *,
        score: float = 1.0,
        success: bool = True,
        mismatch: str | None = None,
        target_contexts: list[ObservedTargetContext] | None = None,
        fail_on_call: int | None = None,
    ):
        self.calls: list[tuple[int, object | None]] = []
        self.score = score
        self.success = success
        self.mismatch = mismatch
        self.target_contexts = target_contexts or []
        self.fail_on_call = fail_on_call

    def run(self, root, interaction_index, *, forced_target=None):
        call_index = len(self.calls)
        self.calls.append((interaction_index, forced_target))
        if self.fail_on_call is not None and call_index + 1 == self.fail_on_call:
            raise RuntimeError("engine invariant violated")
        return ForcedRolloutResult(
            score=None if not self.success else self.score,
            success=self.success,
            mismatch=self.mismatch,
            target_contexts=list(self.target_contexts),
        )


def test_evaluator_runs_exact_root_action_rollout_count() -> None:
    runner = FakeRunner()
    evaluator = RolloutEvaluator(
        FakeInference([9.0]),
        runner,
        root_action_rollouts=10,
        target_contexts_per_action=0,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=1)

    evaluated = evaluator.evaluate_root(root)

    assert len(runner.calls) == 10
    assert all(forced_target is None for _, forced_target in runner.calls)
    assert len(evaluated.action_estimates) == 1
    assert evaluated.action_estimates[0].outcomes.wins == 10


def test_evaluator_aggregates_sampled_actions_by_interaction_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"value": 0}

    def alternating_sample(policy_logits, candidate_indices, **kwargs):
        index = call_count["value"] % len(candidate_indices)
        call_count["value"] += 1
        return candidate_indices[index]

    monkeypatch.setattr(
        "training.policy_iteration.sample_root_action_index",
        alternating_sample,
    )
    runner = FakeRunner(score=1.0)
    evaluator = RolloutEvaluator(
        FakeInference([9.0, 8.0]),
        runner,
        root_action_rollouts=5,
        target_contexts_per_action=0,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=2)

    evaluated = evaluator.evaluate_root(root)

    sampled_indices = {call[0] for call in runner.calls if call[1] is None}
    assert sampled_indices == {0, 1}
    assert {estimate.interaction_index for estimate in evaluated.action_estimates} == {
        0,
        1,
    }
    assert len(evaluated.action_estimates) == 2


def test_evaluator_skips_unsampled_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "training.policy_iteration.sample_root_action_index",
        lambda policy_logits, candidate_indices, **kwargs: 0,
    )
    runner = FakeRunner()
    evaluator = RolloutEvaluator(
        FakeInference([9.0, 1.0]),
        runner,
        root_action_rollouts=10,
        target_contexts_per_action=0,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=2)

    evaluated = evaluator.evaluate_root(root)

    assert {estimate.interaction_index for estimate in evaluated.action_estimates} == {0}
    assert all(call[0] == 0 for call in runner.calls)


def test_evaluator_caps_target_contexts_per_action() -> None:
    contexts = [
        ObservedTargetContext([1], [4, 5], False),
        ObservedTargetContext([2], [6, 7], False),
        ObservedTargetContext([3], [8, 9], False),
    ]
    runner = FakeRunner(target_contexts=contexts)
    evaluator = RolloutEvaluator(
        FakeInference([9.0, 8.0, 7.0]),
        runner,
        root_action_rollouts=1,
        target_contexts_per_action=2,
        target_choices_per_context=1,
        target_rollouts_per_choice=1,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=3)

    evaluated = evaluator.evaluate_root(root)

    assert len(evaluated.target_contexts) == 2
    assert evaluated.target_contexts[0].forced_prefix_deck_ids == [1]
    assert evaluated.target_contexts[1].forced_prefix_deck_ids == [2]


def test_evaluator_runs_one_target_rollout_per_sampled_choice() -> None:
    runner = FakeRunner(
        target_contexts=[ObservedTargetContext([], [4, 5, 6, 7], False)]
    )
    evaluator = RolloutEvaluator(
        FakeInference([9.0]),
        runner,
        root_action_rollouts=1,
        target_contexts_per_action=1,
        target_choices_per_context=3,
        target_rollouts_per_choice=1,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=1)

    evaluated = evaluator.evaluate_root(root)

    target_calls = [call for call in runner.calls if call[1] is not None]
    assert len(target_calls) == 3
    assert len(evaluated.target_contexts) == 1
    assert len(evaluated.target_contexts[0].choices) == 3


def test_evaluator_honors_larger_target_choices_per_context() -> None:
    runner = FakeRunner(
        target_contexts=[ObservedTargetContext([], [4, 5, 6, 7, 8], False)]
    )
    evaluator = RolloutEvaluator(
        FakeInference([9.0]),
        runner,
        root_action_rollouts=1,
        target_contexts_per_action=1,
        target_choices_per_context=5,
        target_rollouts_per_choice=1,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=1)

    evaluated = evaluator.evaluate_root(root)

    target_calls = [call for call in runner.calls if call[1] is not None]
    assert len(target_calls) == 5
    assert len(evaluated.target_contexts[0].choices) == 5


def test_evaluator_counts_mismatches_without_aborting() -> None:
    runner = FakeRunner(success=False, mismatch="Forced target candidate set differs")
    evaluator = RolloutEvaluator(
        FakeInference([9.0]),
        runner,
        root_action_rollouts=2,
        target_contexts_per_action=0,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=1)

    evaluated = evaluator.evaluate_root(root)

    assert evaluated.action_estimates == []
    assert evaluator.stats.mismatched == 2
    assert evaluator.stats.succeeded == 0


def test_evaluator_propagates_engine_exceptions() -> None:
    runner = FakeRunner(fail_on_call=3)
    evaluator = RolloutEvaluator(
        FakeInference([9.0]),
        runner,
        root_action_rollouts=5,
        target_contexts_per_action=0,
        seed=0,
    )
    root = _root(9, "mid_late", interactions=1)

    with pytest.raises(RuntimeError, match="engine invariant violated"):
        evaluator.evaluate_root(root)
