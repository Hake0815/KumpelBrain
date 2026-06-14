"""Seed-state capture and exact forced-action continuation rollouts."""

from __future__ import annotations

import random
import threading
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import Event
from typing import Callable

import torch

from game_player import GamePlayer
from inference_service import InferenceClient, inference_eval_mode
from training.game_state_serialization import is_rollout_root
from training.rollout_data import RootState, phase_budget, stable_root_id


class RolloutMismatch(RuntimeError):
    pass


def match_forced_interaction(
    expected_legal: list[bytes],
    current_legal: list[bytes],
    selected_interaction: bytes,
) -> int:
    del expected_legal
    matches = [
        index
        for index, interaction in enumerate(current_legal)
        if interaction == selected_interaction
    ]
    if not matches:
        raise RolloutMismatch("Forced interaction is no longer legal")
    if len(matches) > 1:
        raise RolloutMismatch("Forced interaction is ambiguous after recreation")
    return matches[0]


def _sample_policy_index(
    logits: torch.Tensor,
    *,
    epsilon: float,
    temperature: float,
    rng: random.Random,
) -> int:
    count = int(logits.numel())
    if count < 1:
        raise ValueError("Cannot sample an empty policy")
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.randrange(count)
    probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=0)
    return int(torch.multinomial(probabilities, 1).item())


class StochasticGamePlayer(GamePlayer):
    def __init__(
        self,
        *args,
        exploration_epsilon: float,
        policy_temperature: float,
        seed: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.exploration_epsilon = exploration_epsilon
        self.policy_temperature = policy_temperature
        self.rng = random.Random(seed)

    def _choose_action_index(self, policy_logits: torch.Tensor) -> int:
        return _sample_policy_index(
            policy_logits,
            epsilon=self.exploration_epsilon,
            temperature=self.policy_temperature,
            rng=self.rng,
        )

    def _choose_target_index(self, policy_logits: torch.Tensor) -> int:
        return _sample_policy_index(
            policy_logits,
            epsilon=self.exploration_epsilon,
            temperature=self.policy_temperature,
            rng=self.rng,
        )


class SeedGamePlayer(StochasticGamePlayer):
    """Play one game and retain each recreatable trainable root state once."""

    def __init__(
        self,
        *args,
        inference_by_player: dict[str, InferenceClient] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.roots: list[RootState] = []
        self.winner_name: str | None = None
        self.inference_by_player = inference_by_player or {}

    def _on_general_update(self, interactions) -> None:
        interaction = interactions[0]
        if interaction.is_game_over():
            self.winner_name = interaction.get_winner_name()
            total = len(self.roots)
            for index, root in enumerate(self.roots):
                root.ply_from_end = total - index - 1
                root.phase = phase_budget(root.ply_from_end).name
            self.inference.game_finished()
            self.callback_on_game_end(interaction.get_game_over_message())
            return
        self._perform_interaction(interaction, None, None, None, "")

    def _on_player_update(self, interactions, player_name: str) -> None:
        previous_inference = self.inference
        previous_device = self.tensor_device
        if player_name in self.inference_by_player:
            self.inference = self.inference_by_player[player_name]
            self.tensor_device = self.inference.tensor_device
        try:
            self._record_and_play_update(interactions, player_name)
        finally:
            self.inference = previous_inference
            self.tensor_device = previous_device

    def _record_and_play_update(self, interactions, player_name: str) -> None:
        if len(interactions) == 1 and not interactions[0].is_with_target():
            self._perform_interaction(interactions[0], None, None, None, player_name)
            return

        state_bytes = self.game_controller.export_game_state_as_byte_array(player_name)
        interaction_bytes = [interaction.to_bytes() for interaction in interactions]
        if is_rollout_root(state_bytes):
            root = RootState(
                root_id=stable_root_id(state_bytes, player_name),
                state_bytes=state_bytes,
                deck_list1=dict(self.deck_list1),
                deck_list2=dict(self.deck_list2),
                player1_name=self.player1_name,
                player2_name=self.player2_name,
                player_name=player_name,
                interaction_bytes=interaction_bytes,
                ply_from_end=-1,
                phase="unknown",
            )
            self.roots.append(root)

        with torch.inference_mode():
            scores, transformed_state, embedded_interactions, card_indices = (
                self.inference.evaluate_move(state_bytes, interaction_bytes)
            )
            chosen_index = (
                0
                if len(interactions) == 1
                else self._choose_action_index(scores.policy_logits)
            )
        self._perform_interaction(
            interactions[chosen_index],
            transformed_state,
            embedded_interactions[chosen_index],
            card_indices,
            player_name,
        )


@dataclass(frozen=True)
class ForcedTargetChoice:
    prefix_deck_ids: tuple[int, ...]
    candidate_deck_ids: tuple[int, ...]
    include_stop_token: bool
    choice_index: int


@dataclass
class ObservedTargetContext:
    prefix_deck_ids: list[int]
    candidate_deck_ids: list[int]
    include_stop_token: bool


@dataclass
class ForcedRolloutResult:
    score: float | None
    success: bool
    error: str | None = None
    target_contexts: list[ObservedTargetContext] = field(default_factory=list)


class ForcedRolloutPlayer(StochasticGamePlayer):
    def __init__(
        self,
        *args,
        root: RootState,
        interaction_index: int,
        forced_target: ForcedTargetChoice | None,
        max_actions: int,
        inference_by_player: dict[str, InferenceClient],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.root = root
        self.expected_interaction = root.interaction_bytes[interaction_index]
        self.forced_target = forced_target
        self.max_actions = max_actions
        self.forced_interaction_pending = True
        self.actions_taken = 0
        self.winner_name: str | None = None
        self.target_contexts: list[ObservedTargetContext] = []
        self.inference_by_player = inference_by_player
        self.mismatch_error: str | None = None

    def _record_mismatch(self, message: str) -> None:
        if self.mismatch_error is None:
            self.mismatch_error = message

    def _on_general_update(self, interactions) -> None:
        interaction = interactions[0]
        if interaction.is_game_over():
            self.winner_name = interaction.get_winner_name()
            self.inference.game_finished()
            self.callback_on_game_end(interaction.get_game_over_message())
            return
        super()._on_general_update(interactions)

    def _on_player_update(self, interactions, player_name: str) -> None:
        previous_inference = self.inference
        previous_device = self.tensor_device
        self.inference = self.inference_by_player[player_name]
        self.tensor_device = self.inference.tensor_device
        try:
            self._handle_player_update(interactions, player_name)
        finally:
            self.inference = previous_inference
            self.tensor_device = previous_device

    def _handle_player_update(self, interactions, player_name: str) -> None:
        if not self.forced_interaction_pending:
            return super()._on_player_update(interactions, player_name)
        if player_name != self.root.player_name:
            self._record_mismatch(
                f"Expected first decision for {self.root.player_name}, got {player_name}"
            )
            self.forced_interaction_pending = False
            return super()._on_player_update(interactions, player_name)
        current_bytes = [interaction.to_bytes() for interaction in interactions]
        try:
            interaction_index = match_forced_interaction(
                self.root.interaction_bytes,
                current_bytes,
                self.expected_interaction,
            )
        except RolloutMismatch as exc:
            self._record_mismatch(str(exc))
            self.forced_interaction_pending = False
            return super()._on_player_update(interactions, player_name)

        state_bytes = self.game_controller.export_game_state_as_byte_array(player_name)
        with torch.inference_mode():
            _, transformed_state, embedded_interactions, card_indices = (
                self.inference.evaluate_move(state_bytes, current_bytes)
            )
        self.forced_interaction_pending = False
        self._perform_interaction(
            interactions[interaction_index],
            transformed_state,
            embedded_interactions[interaction_index],
            card_indices,
            player_name,
        )

    def _perform_interaction(self, *args, **kwargs) -> None:
        self.actions_taken += 1
        if self.actions_taken > self.max_actions:
            self._record_mismatch("Continuation exceeded maximum action count")
            self.callback_on_game_end("Continuation action limit reached")
            return
        super()._perform_interaction(*args, **kwargs)

    def _choose_target_for_context(
        self,
        policy_logits: torch.Tensor,
        candidate_deck_ids: list[int],
        partial_selection_deck_ids: list[int],
        *,
        include_stop_token: bool,
    ) -> int:
        context = ObservedTargetContext(
            prefix_deck_ids=list(partial_selection_deck_ids),
            candidate_deck_ids=list(candidate_deck_ids),
            include_stop_token=include_stop_token,
        )
        self.target_contexts.append(context)
        forced = self.forced_target
        if forced is None:
            return super()._choose_target_for_context(
                policy_logits,
                candidate_deck_ids,
                partial_selection_deck_ids,
                include_stop_token=include_stop_token,
            )

        prefix = tuple(partial_selection_deck_ids)
        if len(prefix) < len(forced.prefix_deck_ids):
            expected_prefix = forced.prefix_deck_ids[: len(prefix)]
            if prefix != expected_prefix:
                self._record_mismatch("Target prefix differs before forced choice")
                return super()._choose_target_for_context(
                    policy_logits,
                    candidate_deck_ids,
                    partial_selection_deck_ids,
                    include_stop_token=include_stop_token,
                )
            next_deck_id = forced.prefix_deck_ids[len(prefix)]
            if next_deck_id not in candidate_deck_ids:
                self._record_mismatch("Forced target prefix card is unavailable")
                return super()._choose_target_for_context(
                    policy_logits,
                    candidate_deck_ids,
                    partial_selection_deck_ids,
                    include_stop_token=include_stop_token,
                )
            return candidate_deck_ids.index(next_deck_id)

        if prefix == forced.prefix_deck_ids:
            if tuple(candidate_deck_ids) != forced.candidate_deck_ids:
                self._record_mismatch("Forced target candidate set differs")
                return super()._choose_target_for_context(
                    policy_logits,
                    candidate_deck_ids,
                    partial_selection_deck_ids,
                    include_stop_token=include_stop_token,
                )
            if include_stop_token != forced.include_stop_token:
                self._record_mismatch("Forced target stop availability differs")
                return super()._choose_target_for_context(
                    policy_logits,
                    candidate_deck_ids,
                    partial_selection_deck_ids,
                    include_stop_token=include_stop_token,
                )
            if forced.choice_index >= len(candidate_deck_ids) + int(include_stop_token):
                self._record_mismatch("Forced target choice index is out of range")
                return super()._choose_target_for_context(
                    policy_logits,
                    candidate_deck_ids,
                    partial_selection_deck_ids,
                    include_stop_token=include_stop_token,
                )
            return forced.choice_index

        return super()._choose_target_for_context(
            policy_logits,
            candidate_deck_ids,
            partial_selection_deck_ids,
            include_stop_token=include_stop_token,
        )


class ContinuationRolloutRunner:
    def __init__(
        self,
        champion: InferenceClient,
        opponent_sampler: Callable[[], InferenceClient],
        compute_device: torch.device,
        *,
        epsilon: float = 0.10,
        temperature: float = 0.5,
        max_actions: int = 512,
        timeout_s: float = 120.0,
        seed: int = 0,
    ):
        self.champion = champion
        self.opponent_sampler = opponent_sampler
        self.compute_device = compute_device
        self.epsilon = epsilon
        self.temperature = temperature
        self.max_actions = max_actions
        self.timeout_s = timeout_s
        self.rng = random.Random(seed)

    def run(
        self,
        root: RootState,
        interaction_index: int,
        *,
        forced_target: ForcedTargetChoice | None = None,
    ) -> ForcedRolloutResult:
        opponent = self.opponent_sampler()
        inference_by_player = {
            root.player_name: self.champion,
            (
                root.player2_name
                if root.player_name == root.player1_name
                else root.player1_name
            ): opponent,
        }
        done = Event()
        player = ForcedRolloutPlayer(
            deck_list1=root.deck_list1,
            deck_list2=root.deck_list2,
            player1_name=root.player1_name,
            player2_name=root.player2_name,
            game_uuid=uuid.uuid4(),
            callback_on_game_end=lambda _message: done.set(),
            inference=self.champion,
            compute_device=self.compute_device,
            exploration_epsilon=self.epsilon,
            policy_temperature=self.temperature,
            seed=self.rng.randrange(2**31),
            root=root,
            interaction_index=interaction_index,
            forced_target=forced_target,
            max_actions=self.max_actions,
            inference_by_player=inference_by_player,
        )
        try:
            error_holder: list[BaseException] = []

            def play() -> None:
                try:
                    with ExitStack() as stack:
                        stack.enter_context(inference_eval_mode(self.champion))
                        if opponent is not self.champion:
                            stack.enter_context(inference_eval_mode(opponent))
                        player.play_from_state(root.state_bytes)
                except BaseException as exc:
                    error_holder.append(exc)
                    done.set()

            thread = threading.Thread(
                target=play,
                name="forced-rollout",
                daemon=True,
            )
            thread.start()
            if not done.wait(timeout=self.timeout_s):
                raise TimeoutError("Continuation timed out")
            thread.join(timeout=1.0)
            if error_holder:
                raise error_holder[0]
            if player.mismatch_error is not None:
                raise RolloutMismatch(player.mismatch_error)
            if player.winner_name is None:
                score = 0.5
            else:
                score = 1.0 if player.winner_name == root.player_name else 0.0
            return ForcedRolloutResult(
                score=score,
                success=True,
                target_contexts=player.target_contexts,
            )
        except Exception as exc:  # noqa: BLE001 - rollout quarantine boundary
            self.champion.game_finished()
            if opponent is not self.champion:
                opponent.game_finished()
            return ForcedRolloutResult(
                score=None,
                success=False,
                error=str(exc),
                target_contexts=player.target_contexts,
            )
