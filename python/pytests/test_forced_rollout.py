"""Exact forced interaction and hierarchical target-choice tests."""

from __future__ import annotations

import random
import sys
import threading
import time
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

from training.rollout_runner import (
    ContinuationRolloutRunner,
    ForcedRolloutPlayer,
    ForcedTargetChoice,
    RolloutMismatch,
    match_forced_interaction,
)
from training.rollout_data import RootState


def test_forced_interaction_requires_selected_action_to_remain_unique() -> None:
    assert match_forced_interaction([b"a", b"b"], [b"a", b"b"], b"b") == 1
    assert match_forced_interaction([b"a", b"b"], [b"x", b"b"], b"b") == 1
    with pytest.raises(RolloutMismatch, match="no longer legal"):
        match_forced_interaction([b"a", b"b"], [b"x", b"y"], b"b")
    with pytest.raises(RolloutMismatch, match="ambiguous"):
        match_forced_interaction([b"a", b"b"], [b"b", b"b"], b"b")


def _player(forced: ForcedTargetChoice) -> ForcedRolloutPlayer:
    player = object.__new__(ForcedRolloutPlayer)
    player.forced_target = forced
    player.target_contexts = []
    player.mismatch_error = None
    player.exploration_epsilon = 0.0
    player.policy_temperature = 0.5
    player.rng = random.Random(0)
    return player


def test_forced_target_replays_prefix_then_selected_candidate() -> None:
    player = _player(
        ForcedTargetChoice(
            prefix_deck_ids=(7,),
            candidate_deck_ids=(4, 9),
            include_stop_token=False,
            choice_index=1,
        )
    )
    logits = torch.tensor([0.0, 0.0])

    prefix_choice = player._choose_target_for_context(
        logits,
        [7, 8],
        [],
        include_stop_token=False,
    )
    forced_choice = player._choose_target_for_context(
        logits,
        [4, 9],
        [7],
        include_stop_token=False,
    )

    assert prefix_choice == 0
    assert forced_choice == 1


def test_forced_stop_token_is_supported() -> None:
    player = _player(
        ForcedTargetChoice(
            prefix_deck_ids=(7,),
            candidate_deck_ids=(4, 9),
            include_stop_token=True,
            choice_index=2,
        )
    )

    choice = player._choose_target_for_context(
        torch.zeros(3),
        [4, 9],
        [7],
        include_stop_token=True,
    )

    assert choice == 2


def test_forced_target_candidate_mismatch_is_recorded_without_callback_error() -> None:
    player = _player(
        ForcedTargetChoice(
            prefix_deck_ids=(),
            candidate_deck_ids=(4, 9),
            include_stop_token=False,
            choice_index=0,
        )
    )

    choice = player._choose_target_for_context(
        torch.zeros(2),
        [4, 8],
        [],
        include_stop_token=False,
    )

    assert choice in {0, 1}
    assert player.mismatch_error == "Forced target candidate set differs"


def test_continuation_waits_for_terminal_callback(monkeypatch) -> None:
    class FakeInference:
        def game_finished(self) -> None:
            pass

    class AsyncFakePlayer:
        def __init__(self, *args, callback_on_game_end, **kwargs):
            self.callback_on_game_end = callback_on_game_end
            self.winner_name = None
            self.mismatch_error = None
            self.target_contexts = []

        def play_from_state(self, state_bytes: bytes) -> None:
            def finish() -> None:
                self.winner_name = "player1"
                self.callback_on_game_end("finished")

            threading.Timer(0.1, finish).start()

    monkeypatch.setattr(
        "training.rollout_runner.ForcedRolloutPlayer",
        AsyncFakePlayer,
    )
    inference = FakeInference()
    root = RootState(
        root_id="root",
        state_bytes=b"state",
        deck_list1={},
        deck_list2={},
        player1_name="player1",
        player2_name="player2",
        player_name="player1",
        interaction_bytes=[b"interaction"],
        ply_from_end=0,
        phase="terminal",
    )
    runner = ContinuationRolloutRunner(
        inference,
        lambda: inference,
        torch.device("cpu"),
        timeout_s=1.0,
    )

    started = time.monotonic()
    result = runner.run(root, 0)

    assert time.monotonic() - started >= 0.08
    assert result.success
    assert result.score == 1.0


def test_continuation_propagates_player_exceptions(monkeypatch) -> None:
    class FakeInference:
        def game_finished(self) -> None:
            pass

    class FailingFakePlayer:
        def __init__(self, *args, **kwargs):
            self.winner_name = None
            self.mismatch_error = None
            self.target_contexts = []

        def play_from_state(self, state_bytes: bytes) -> None:
            raise RuntimeError("engine invariant violated")

    monkeypatch.setattr(
        "training.rollout_runner.ForcedRolloutPlayer",
        FailingFakePlayer,
    )
    inference = FakeInference()
    root = RootState(
        root_id="root",
        state_bytes=b"state",
        deck_list1={},
        deck_list2={},
        player1_name="player1",
        player2_name="player2",
        player_name="player1",
        interaction_bytes=[b"interaction"],
        ply_from_end=0,
        phase="terminal",
    )
    runner = ContinuationRolloutRunner(
        inference,
        lambda: inference,
        torch.device("cpu"),
        timeout_s=1.0,
    )

    with pytest.raises(RuntimeError, match="engine invariant violated"):
        runner.run(root, 0)


def test_continuation_returns_replay_mismatch_as_missing_observation(
    monkeypatch,
) -> None:
    class FakeInference:
        def game_finished(self) -> None:
            pass

    class MismatchedFakePlayer:
        def __init__(self, *args, callback_on_game_end, **kwargs):
            self.callback_on_game_end = callback_on_game_end
            self.winner_name = "player1"
            self.mismatch_error = "Forced target candidate set differs"
            self.target_contexts = []

        def play_from_state(self, state_bytes: bytes) -> None:
            self.callback_on_game_end("finished")

    monkeypatch.setattr(
        "training.rollout_runner.ForcedRolloutPlayer",
        MismatchedFakePlayer,
    )
    inference = FakeInference()
    root = RootState(
        root_id="root",
        state_bytes=b"state",
        deck_list1={},
        deck_list2={},
        player1_name="player1",
        player2_name="player2",
        player_name="player1",
        interaction_bytes=[b"interaction"],
        ply_from_end=0,
        phase="terminal",
    )
    runner = ContinuationRolloutRunner(
        inference,
        lambda: inference,
        torch.device("cpu"),
        timeout_s=1.0,
    )

    result = runner.run(root, 0)

    assert not result.success
    assert result.score is None
    assert result.mismatch == "Forced target candidate set differs"
