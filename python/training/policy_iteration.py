"""Rollout branch evaluation, opponent sampling, and champion gating."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from inference_models import create_self_play_models
from inference_service import DirectInferenceClient, InferenceClient
from training.rollout_data import (
    ActionEstimate,
    RootState,
    TargetChoiceEstimate,
    TargetContextEstimate,
    phase_budget,
    select_branches,
    select_target_branches,
)
from training.rollout_runner import (
    ContinuationRolloutRunner,
    ForcedTargetChoice,
    ObservedTargetContext,
)
from training.trainer import ModelStore


def sample_phase_balanced_roots(
    roots: list[RootState],
    count: int,
    *,
    seed: int,
) -> list[RootState]:
    rng = random.Random(seed)
    buckets: dict[str, list[RootState]] = {}
    for root in roots:
        buckets.setdefault(root.phase, []).append(root)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[RootState] = []
    phase_names = [budget.name for budget in (
        phase_budget(0),
        phase_budget(3),
        phase_budget(8),
        phase_budget(16),
        phase_budget(32),
    )]
    while len(selected) < count:
        added = False
        for phase_name in phase_names:
            bucket = buckets.get(phase_name, [])
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
                added = True
        if not added:
            break
    return selected


@dataclass
class RolloutEvaluationStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def record(self, success: bool, error: str | None) -> None:
        self.attempted += 1
        if success:
            self.succeeded += 1
        else:
            self.failed += 1
            if error:
                self.errors.append(error)


class RolloutEvaluator:
    def __init__(
        self,
        inference: InferenceClient,
        runner: ContinuationRolloutRunner,
        *,
        seed: int = 0,
    ):
        self.inference = inference
        self.runner = runner
        self.rng = random.Random(seed)
        self.stats = RolloutEvaluationStats()

    def evaluate_root(
        self,
        root: RootState,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> RootState:
        with torch.inference_mode():
            scores, transformed_state, embedded_interactions, card_indices = (
                self.inference.evaluate_move(root.state_bytes, root.interaction_bytes)
            )
        budget = phase_budget(root.ply_from_end)
        action_indices = select_branches(
            scores.policy_logits.detach().cpu(),
            budget,
            rng=self.rng,
        )
        observed_contexts: dict[
            tuple[int, tuple[int, ...], tuple[int, ...], bool],
            ObservedTargetContext,
        ] = {}
        root.action_estimates = []
        root.target_contexts = []

        for interaction_index in action_indices:
            estimate = ActionEstimate(interaction_index)
            for _ in range(budget.rollouts_per_action):
                result = self.runner.run(root, interaction_index)
                self.stats.record(result.success, result.error)
                if result.success and result.score is not None:
                    estimate.outcomes.add(result.score)
                for context in result.target_contexts:
                    key = (
                        interaction_index,
                        tuple(context.prefix_deck_ids),
                        tuple(context.candidate_deck_ids),
                        context.include_stop_token,
                    )
                    observed_contexts[key] = context
                if progress is not None:
                    progress(1)
            if estimate.outcomes.total:
                root.action_estimates.append(estimate)

        for key, context in observed_contexts.items():
            interaction_index = key[0]
            candidates = torch.tensor(
                context.candidate_deck_ids,
                device=transformed_state.device,
                dtype=torch.long,
            )
            prefix = torch.tensor(
                context.prefix_deck_ids,
                device=transformed_state.device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                target_scores = self.inference.score_targets(
                    candidates,
                    prefix,
                    transformed_state,
                    embedded_interactions[interaction_index],
                    card_indices,
                    context.include_stop_token,
                )
            choice_indices = select_target_branches(
                target_scores.policy_logits.detach().cpu(),
                rng=self.rng,
            )
            target_context = TargetContextEstimate(
                interaction_index=interaction_index,
                forced_prefix_deck_ids=list(context.prefix_deck_ids),
                candidate_deck_ids=list(context.candidate_deck_ids),
                include_stop_token=context.include_stop_token,
            )
            target_rollouts = min(budget.rollouts_per_action, 4)
            for choice_index in choice_indices:
                choice = TargetChoiceEstimate(choice_index)
                forced_target = ForcedTargetChoice(
                    prefix_deck_ids=tuple(context.prefix_deck_ids),
                    candidate_deck_ids=tuple(context.candidate_deck_ids),
                    include_stop_token=context.include_stop_token,
                    choice_index=choice_index,
                )
                for _ in range(target_rollouts):
                    result = self.runner.run(
                        root,
                        interaction_index,
                        forced_target=forced_target,
                    )
                    self.stats.record(result.success, result.error)
                    if result.success and result.score is not None:
                        choice.outcomes.add(result.score)
                    if progress is not None:
                        progress(1)
                if choice.outcomes.total:
                    target_context.choices.append(choice)
            if target_context.choices:
                root.target_contexts.append(target_context)
        return root


class OpponentPool:
    """Frozen champion/history clients sampled with a 75/25 mixture."""

    def __init__(
        self,
        model_store: ModelStore,
        device: torch.device,
        *,
        history_limit: int = 4,
        seed: int = 0,
    ):
        self.rng = random.Random(seed)
        self.champion = self._load_client(model_store.champion_path, device)
        history_paths = sorted(model_store.history_dir.glob("champion_*.pt"))[
            -history_limit:
        ]
        self.history = [self._load_client(path, device) for path in history_paths]

    @staticmethod
    def _load_client(path: Path, device: torch.device) -> DirectInferenceClient:
        network, selector, _ = create_self_play_models(
            device,
            embed_on_compute_device=device.type == "cuda",
        )
        ModelStore.load_checkpoint(path, network, selector, device)
        network.eval()
        selector.eval()
        return DirectInferenceClient(network, selector, device)

    def sample(
        self,
        champion_override: InferenceClient | None = None,
    ) -> InferenceClient:
        if self.history and self.rng.random() >= 0.75:
            return self.rng.choice(self.history)
        return champion_override or self.champion


@dataclass(frozen=True)
class WilsonInterval:
    lower: float
    upper: float


def wilson_interval(
    score_sum: float,
    games: int,
    *,
    z: float = 1.2815515655446004,
) -> WilsonInterval:
    if games < 1:
        return WilsonInterval(0.0, 1.0)
    proportion = score_sum / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / games
            + z * z / (4.0 * games * games)
        )
        / denominator
    )
    return WilsonInterval(max(0.0, center - spread), min(1.0, center + spread))


@dataclass(frozen=True)
class GateResult:
    promoted: bool
    games: int
    wins: int
    draws: int
    losses: int
    interval: WilsonInterval
    reason: str


def run_champion_gate(
    play_block: Callable[[int, bool], list[float]],
    *,
    block_size: int = 100,
    minimum_games: int = 200,
    maximum_games: int = 800,
) -> GateResult:
    wins = draws = losses = 0
    games = 0
    interval = WilsonInterval(0.0, 1.0)
    while games < maximum_games:
        remaining = min(block_size, maximum_games - games)
        first_side = remaining // 2
        scores = play_block(first_side, True) + play_block(
            remaining - first_side,
            False,
        )
        for score in scores:
            if score == 1.0:
                wins += 1
            elif score == 0.5:
                draws += 1
            elif score == 0.0:
                losses += 1
            else:
                raise ValueError(f"Invalid gate score: {score}")
        games += len(scores)
        interval = wilson_interval(wins + 0.5 * draws, games)
        if games >= minimum_games and interval.lower > 0.5:
            return GateResult(
                True, games, wins, draws, losses, interval, "lower bound exceeds 0.5"
            )
        if games >= minimum_games and interval.upper <= 0.5:
            return GateResult(
                False, games, wins, draws, losses, interval, "upper bound is at most 0.5"
            )
    return GateResult(
        False,
        games,
        wins,
        draws,
        losses,
        interval,
        "maximum games reached without a conclusive result",
    )
