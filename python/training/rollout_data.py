"""Rollout-policy-iteration records, storage, and target construction."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

POLICY_TEMPERATURE = 0.25
PROBABILITY_EPSILON = 1e-4


@dataclass
class OutcomeCounts:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    def add(self, score: float) -> None:
        if score == 1.0:
            self.wins += 1
        elif score == 0.5:
            self.draws += 1
        elif score == 0.0:
            self.losses += 1
        else:
            raise ValueError(f"Outcome score must be 0, 0.5, or 1; got {score}")

    @property
    def value_target(self) -> float:
        return (self.wins + 0.5 * self.draws + 1.0) / (self.total + 2.0)

    @property
    def confidence_weight(self) -> float:
        return self.total / (self.total + 2.0)


@dataclass
class ActionEstimate:
    interaction_index: int
    outcomes: OutcomeCounts = field(default_factory=OutcomeCounts)


@dataclass
class TargetChoiceEstimate:
    choice_index: int
    outcomes: OutcomeCounts = field(default_factory=OutcomeCounts)


@dataclass
class TargetContextEstimate:
    interaction_index: int
    forced_prefix_deck_ids: list[int]
    candidate_deck_ids: list[int]
    include_stop_token: bool
    choices: list[TargetChoiceEstimate] = field(default_factory=list)


@dataclass
class RootState:
    root_id: str
    state_bytes: bytes
    deck_list1: dict[str, int]
    deck_list2: dict[str, int]
    player1_name: str
    player2_name: str
    player_name: str
    interaction_bytes: list[bytes]
    ply_from_end: int
    phase: str
    action_estimates: list[ActionEstimate] = field(default_factory=list)
    target_contexts: list[TargetContextEstimate] = field(default_factory=list)


@dataclass
class RolloutShard:
    iteration: int
    champion_id: str
    roots: list[RootState]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseBudget:
    name: str
    min_ply_from_end: int
    max_ply_from_end: int | None
    top_actions: int | None
    random_actions: int
    max_actions: int | None
    rollouts_per_action: int

    def contains(self, ply_from_end: int) -> bool:
        return (
            ply_from_end >= self.min_ply_from_end
            and (
                self.max_ply_from_end is None
                or ply_from_end <= self.max_ply_from_end
            )
        )


PHASE_BUDGETS = (
    PhaseBudget("terminal", 0, 2, None, 0, None, 16),
    PhaseBudget("late", 3, 7, None, 0, 8, 8),
    PhaseBudget("mid_late", 8, 15, 4, 2, 6, 4),
    PhaseBudget("mid", 16, 31, 3, 1, 4, 2),
    PhaseBudget("early", 32, None, 3, 1, 4, 2),
)


def phase_budget(ply_from_end: int) -> PhaseBudget:
    return next(budget for budget in PHASE_BUDGETS if budget.contains(ply_from_end))


def stable_root_id(state_bytes: bytes, player_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(state_bytes)
    digest.update(b"\0")
    digest.update(player_name.encode("utf-8"))
    return digest.hexdigest()


def is_validation_root(root_id: str, fraction: float = 0.1) -> bool:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("validation fraction must be within [0, 1]")
    bucket = int(root_id[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return bucket < fraction


def select_branches(
    policy_logits: torch.Tensor,
    budget: PhaseBudget,
    *,
    rng: random.Random,
) -> list[int]:
    count = int(policy_logits.numel())
    if count == 0:
        return []
    if budget.top_actions is None:
        indices = list(range(count))
        if budget.max_actions is not None and len(indices) > budget.max_actions:
            ranked = torch.argsort(policy_logits, descending=True).tolist()
            return [int(index) for index in ranked[: budget.max_actions]]
        return indices

    top_count = min(budget.top_actions, count)
    ranked = [int(index) for index in torch.argsort(
        policy_logits, descending=True
    ).tolist()]
    selected = ranked[:top_count]
    remaining = [index for index in range(count) if index not in selected]
    selected.extend(rng.sample(remaining, k=min(budget.random_actions, len(remaining))))
    return selected


def select_target_branches(
    policy_logits: torch.Tensor,
    *,
    rng: random.Random,
) -> list[int]:
    count = int(policy_logits.numel())
    if count <= 8:
        return list(range(count))
    budget = PhaseBudget("target", 0, None, 4, 2, 6, 0)
    return select_branches(policy_logits, budget, rng=rng)


def candidate_action_indices(
    policy_logits: torch.Tensor,
    budget: PhaseBudget,
    *,
    rng: random.Random,
) -> list[int]:
    """Return phase-capped legal interaction indices used as a sampling mask."""
    return select_branches(policy_logits, budget, rng=rng)


def sample_root_action_index(
    policy_logits: torch.Tensor,
    candidate_indices: list[int],
    *,
    temperature: float,
    rng: random.Random,
) -> int:
    if not candidate_indices:
        raise ValueError("At least one candidate action is required")
    if len(candidate_indices) == 1:
        return candidate_indices[0]
    candidate_logits = policy_logits[candidate_indices] / max(temperature, 1e-6)
    probabilities = torch.softmax(candidate_logits, dim=0).tolist()
    sampled = rng.choices(range(len(candidate_indices)), weights=probabilities, k=1)[0]
    return candidate_indices[sampled]


def select_sampled_target_branches(
    policy_logits: torch.Tensor,
    *,
    rng: random.Random,
    top_count: int = 2,
    random_count: int = 1,
    max_choices: int | None = None,
) -> list[int]:
    count = int(policy_logits.numel())
    if count == 0:
        return []
    limit = count if max_choices is None else min(max_choices, count)
    if count <= limit:
        return list(range(count))
    ranked = [
        int(index)
        for index in torch.argsort(policy_logits, descending=True).tolist()
    ]
    selected = ranked[: min(top_count, limit)]
    remaining = [index for index in range(count) if index not in selected]
    selected.extend(
        rng.sample(remaining, k=min(random_count, limit - len(selected), len(remaining)))
    )
    return selected[:limit]


def improved_policy(
    estimates: list[ActionEstimate] | list[TargetChoiceEstimate],
) -> torch.Tensor:
    if not estimates:
        return torch.empty(0)
    probabilities = torch.tensor(
        [estimate.outcomes.value_target for estimate in estimates],
        dtype=torch.float32,
    ).clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    logits = torch.logit(probabilities) / POLICY_TEMPERATURE
    return torch.softmax(logits, dim=0)


class RolloutStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, shard: RolloutShard) -> Path:
        path = self.root / f"rollout_{shard.iteration:06d}.pt"
        torch.save(asdict(shard), path)
        return path

    def load(self, path: str | Path) -> RolloutShard:
        payload = torch.load(path, weights_only=False)
        return _shard_from_dict(payload)

    def recent(self, count: int) -> list[RolloutShard]:
        paths = sorted(self.root.glob("rollout_*.pt"))[-count:]
        return [self.load(path) for path in paths]


def _outcomes(payload: dict[str, Any]) -> OutcomeCounts:
    return OutcomeCounts(
        wins=int(payload["wins"]),
        draws=int(payload["draws"]),
        losses=int(payload["losses"]),
    )


def _shard_from_dict(payload: dict[str, Any]) -> RolloutShard:
    roots: list[RootState] = []
    for item in payload["roots"]:
        action_estimates = [
            ActionEstimate(
                interaction_index=int(estimate["interaction_index"]),
                outcomes=_outcomes(estimate["outcomes"]),
            )
            for estimate in item["action_estimates"]
        ]
        target_contexts = [
            TargetContextEstimate(
                interaction_index=int(context["interaction_index"]),
                forced_prefix_deck_ids=[
                    int(value) for value in context["forced_prefix_deck_ids"]
                ],
                candidate_deck_ids=[
                    int(value) for value in context["candidate_deck_ids"]
                ],
                include_stop_token=bool(context["include_stop_token"]),
                choices=[
                    TargetChoiceEstimate(
                        choice_index=int(choice["choice_index"]),
                        outcomes=_outcomes(choice["outcomes"]),
                    )
                    for choice in context["choices"]
                ],
            )
            for context in item["target_contexts"]
        ]
        roots.append(
            RootState(
                root_id=item["root_id"],
                state_bytes=item["state_bytes"],
                deck_list1=dict(item["deck_list1"]),
                deck_list2=dict(item["deck_list2"]),
                player1_name=item["player1_name"],
                player2_name=item["player2_name"],
                player_name=item["player_name"],
                interaction_bytes=list(item["interaction_bytes"]),
                ply_from_end=int(item["ply_from_end"]),
                phase=item["phase"],
                action_estimates=action_estimates,
                target_contexts=target_contexts,
            )
        )
    return RolloutShard(
        iteration=int(payload["iteration"]),
        champion_id=payload["champion_id"],
        roots=roots,
        metadata=dict(payload["metadata"]),
    )
