"""Dual-head rollout training and fresh checkpoint management."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from network.kumpel_network import KumpelNetwork
from network.selector import Selector
from training.rollout_data import (
    RootState,
    RolloutShard,
    improved_policy,
    is_validation_root,
)


@dataclass(frozen=True)
class LossComponents:
    total: torch.Tensor
    interaction_value: torch.Tensor
    interaction_policy: torch.Tensor
    target_value: torch.Tensor | None
    target_policy: torch.Tensor | None


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    interaction_value: float
    interaction_policy: float
    target_value: float | None
    target_policy: float | None


@dataclass(frozen=True)
class CheckpointMetadata:
    iteration: int
    champion_id: str
    role: str


def split_roots(
    shards: list[RolloutShard],
    *,
    validation_fraction: float = 0.1,
) -> tuple[list[RootState], list[RootState]]:
    roots = [root for shard in shards for root in shard.roots]
    training = [
        root
        for root in roots
        if not is_validation_root(root.root_id, validation_fraction)
    ]
    validation = [
        root
        for root in roots
        if is_validation_root(root.root_id, validation_fraction)
    ]
    return training, validation


class RolloutTrainer:
    def __init__(
        self,
        network: KumpelNetwork,
        selector: Selector,
        device: torch.device,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        seed: int = 0,
    ):
        self.network = network
        self.selector = selector
        self.device = device
        self.gradient_clip = gradient_clip
        self.rng = random.Random(seed)
        self.optimizer = torch.optim.AdamW(
            list(network.parameters()) + list(selector.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.train_steps = 0

    @staticmethod
    def _weighted_value_loss(
        logits: list[torch.Tensor],
        targets: list[float],
        weights: list[float],
    ) -> torch.Tensor:
        stacked_logits = torch.stack(logits)
        target_tensor = torch.tensor(
            targets,
            device=stacked_logits.device,
            dtype=stacked_logits.dtype,
        )
        weight_tensor = torch.tensor(
            weights,
            device=stacked_logits.device,
            dtype=stacked_logits.dtype,
        )
        losses = F.binary_cross_entropy_with_logits(
            stacked_logits,
            target_tensor,
            reduction="none",
        )
        return (losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)

    @staticmethod
    def _policy_loss(
        logits_per_context: list[torch.Tensor],
        targets_per_context: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        for logits, targets in zip(
            logits_per_context,
            targets_per_context,
            strict=True,
        ):
            targets = targets.to(device=logits.device, dtype=logits.dtype)
            losses.append(-(targets * F.log_softmax(logits, dim=0)).sum())
        return torch.stack(losses).mean()

    def losses(self, roots: list[RootState]) -> LossComponents:
        if not roots:
            raise ValueError("At least one rollout root is required")
        (
            interaction_scores,
            transformed_state,
            embedded_interactions,
            card_indices,
            interaction_mask,
            state_mask,
        ) = self.network.forward_batch(
            [root.state_bytes for root in roots],
            [root.interaction_bytes for root in roots],
        )

        interaction_value_logits: list[torch.Tensor] = []
        interaction_value_targets: list[float] = []
        interaction_value_weights: list[float] = []
        interaction_policy_logits: list[torch.Tensor] = []
        interaction_policy_targets: list[torch.Tensor] = []
        target_value_logits: list[torch.Tensor] = []
        target_value_targets: list[float] = []
        target_value_weights: list[float] = []
        target_policy_logits: list[torch.Tensor] = []
        target_policy_targets: list[torch.Tensor] = []

        for batch_index, root in enumerate(roots):
            valid_interactions = interaction_mask[batch_index]
            action_indices = [
                estimate.interaction_index for estimate in root.action_estimates
            ]
            if action_indices:
                value_logits = interaction_scores.value_logits[
                    batch_index, valid_interactions
                ]
                policy_logits = interaction_scores.policy_logits[
                    batch_index, valid_interactions
                ]
                interaction_value_logits.extend(
                    value_logits[index] for index in action_indices
                )
                interaction_value_targets.extend(
                    estimate.outcomes.value_target
                    for estimate in root.action_estimates
                )
                interaction_value_weights.extend(
                    estimate.outcomes.confidence_weight
                    for estimate in root.action_estimates
                )
                interaction_policy_logits.append(policy_logits[action_indices])
                interaction_policy_targets.append(improved_policy(root.action_estimates))

            valid_state = transformed_state[batch_index, state_mask[batch_index]]
            valid_embedded_interactions = embedded_interactions[
                batch_index, valid_interactions
            ]
            for context in root.target_contexts:
                candidates = torch.tensor(
                    context.candidate_deck_ids,
                    device=self.device,
                    dtype=torch.long,
                )
                prefix = torch.tensor(
                    context.forced_prefix_deck_ids,
                    device=self.device,
                    dtype=torch.long,
                )
                scores = self.selector(
                    candidates,
                    prefix,
                    valid_state,
                    valid_embedded_interactions[context.interaction_index],
                    card_indices[batch_index],
                    include_stop_token=context.include_stop_token,
                )
                choice_indices = [choice.choice_index for choice in context.choices]
                target_value_logits.extend(
                    scores.value_logits[index] for index in choice_indices
                )
                target_value_targets.extend(
                    choice.outcomes.value_target for choice in context.choices
                )
                target_value_weights.extend(
                    choice.outcomes.confidence_weight for choice in context.choices
                )
                target_policy_logits.append(scores.policy_logits[choice_indices])
                target_policy_targets.append(improved_policy(context.choices))

        if not interaction_value_logits or not interaction_policy_logits:
            raise ValueError("Rollout roots contain no evaluated interactions")

        interaction_value = self._weighted_value_loss(
            interaction_value_logits,
            interaction_value_targets,
            interaction_value_weights,
        )
        interaction_policy = self._policy_loss(
            interaction_policy_logits,
            interaction_policy_targets,
        )
        target_value = (
            self._weighted_value_loss(
                target_value_logits,
                target_value_targets,
                target_value_weights,
            )
            if target_value_logits
            else None
        )
        target_policy = (
            self._policy_loss(target_policy_logits, target_policy_targets)
            if target_policy_logits
            else None
        )
        total = interaction_value + interaction_policy
        if target_value is not None:
            total = total + target_value
        if target_policy is not None:
            total = total + target_policy
        return LossComponents(
            total,
            interaction_value,
            interaction_policy,
            target_value,
            target_policy,
        )

    def train_step(self, roots: list[RootState]) -> TrainStepResult:
        self.network.train()
        self.selector.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.losses(roots)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.network.parameters()) + list(self.selector.parameters()),
            self.gradient_clip,
        )
        self.optimizer.step()
        self.train_steps += 1
        return TrainStepResult(
            loss=float(losses.total.item()),
            interaction_value=float(losses.interaction_value.item()),
            interaction_policy=float(losses.interaction_policy.item()),
            target_value=(
                None if losses.target_value is None else float(losses.target_value.item())
            ),
            target_policy=(
                None
                if losses.target_policy is None
                else float(losses.target_policy.item())
            ),
        )

    def validation_loss(self, roots: list[RootState], batch_size: int) -> float:
        self.network.eval()
        self.selector.eval()
        weighted_loss = 0.0
        root_count = 0
        with torch.inference_mode():
            for start in range(0, len(roots), batch_size):
                batch = roots[start : start + batch_size]
                loss = self.losses(batch).total
                weighted_loss += float(loss.item()) * len(batch)
                root_count += len(batch)
        return weighted_loss / max(root_count, 1)

    def calibration_metrics(
        self,
        roots: list[RootState],
        batch_size: int,
    ) -> dict[str, float | None]:
        self.network.eval()
        self.selector.eval()
        interaction_squared_error = 0.0
        interaction_count = 0
        target_squared_error = 0.0
        target_count = 0
        with torch.inference_mode():
            for start in range(0, len(roots), batch_size):
                batch = roots[start : start + batch_size]
                (
                    interaction_scores,
                    transformed_state,
                    embedded_interactions,
                    card_indices,
                    interaction_mask,
                    state_mask,
                ) = self.network.forward_batch(
                    [root.state_bytes for root in batch],
                    [root.interaction_bytes for root in batch],
                )
                for batch_index, root in enumerate(batch):
                    valid_interactions = interaction_mask[batch_index]
                    values = torch.sigmoid(
                        interaction_scores.value_logits[
                            batch_index, valid_interactions
                        ]
                    )
                    for estimate in root.action_estimates:
                        error = (
                            float(values[estimate.interaction_index].item())
                            - estimate.outcomes.value_target
                        )
                        interaction_squared_error += error * error
                        interaction_count += 1

                    valid_state = transformed_state[
                        batch_index, state_mask[batch_index]
                    ]
                    valid_embedded_interactions = embedded_interactions[
                        batch_index, valid_interactions
                    ]
                    for context in root.target_contexts:
                        scores = self.selector(
                            torch.tensor(
                                context.candidate_deck_ids,
                                device=self.device,
                                dtype=torch.long,
                            ),
                            torch.tensor(
                                context.forced_prefix_deck_ids,
                                device=self.device,
                                dtype=torch.long,
                            ),
                            valid_state,
                            valid_embedded_interactions[context.interaction_index],
                            card_indices[batch_index],
                            include_stop_token=context.include_stop_token,
                        )
                        values = torch.sigmoid(scores.value_logits)
                        for choice in context.choices:
                            error = (
                                float(values[choice.choice_index].item())
                                - choice.outcomes.value_target
                            )
                            target_squared_error += error * error
                            target_count += 1
        return {
            "interaction_brier": (
                interaction_squared_error / interaction_count
                if interaction_count
                else None
            ),
            "target_brier": (
                target_squared_error / target_count if target_count else None
            ),
        }

    def sample_batch(self, roots: list[RootState], batch_size: int) -> list[RootState]:
        if len(roots) <= batch_size:
            return list(roots)
        return self.rng.sample(roots, batch_size)


class ModelStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.champion_path = self.root / "champion.pt"
        self.candidate_path = self.root / "candidate.last.pt"
        self.history_dir = self.root / "history"
        self.history_dir.mkdir(exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def save_checkpoint(
        path: str | Path,
        network: KumpelNetwork,
        selector: Selector,
        *,
        metadata: CheckpointMetadata,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "rollout-policy-v1",
                "network": network.state_dict(),
                "selector": selector.state_dict(),
                "metadata": {
                    "iteration": metadata.iteration,
                    "champion_id": metadata.champion_id,
                    "role": metadata.role,
                },
                "metrics": metrics or {},
            },
            path,
        )

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        network: KumpelNetwork,
        selector: Selector,
        device: torch.device,
    ) -> CheckpointMetadata:
        payload = torch.load(path, weights_only=False, map_location=device)
        if payload.get("format") != "rollout-policy-v1":
            raise ValueError(f"Unsupported checkpoint format: {path}")
        network.load_state_dict(payload["network"])
        selector.load_state_dict(payload["selector"])
        metadata = payload["metadata"]
        return CheckpointMetadata(
            iteration=int(metadata["iteration"]),
            champion_id=metadata["champion_id"],
            role=metadata["role"],
        )

    def promote(
        self,
        candidate_path: str | Path,
        *,
        previous_champion_id: str,
        iteration: int,
        history_limit: int,
        manifest: dict[str, Any],
    ) -> None:
        if self.champion_path.exists():
            history_path = (
                self.history_dir
                / f"champion_{iteration - 1:06d}_{previous_champion_id}.pt"
            )
            shutil.copy2(self.champion_path, history_path)

        with tempfile.NamedTemporaryFile(
            dir=self.root,
            prefix="champion.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            payload = torch.load(
                candidate_path,
                weights_only=False,
                map_location="cpu",
            )
            payload["metadata"]["role"] = "champion"
            torch.save(payload, temporary)
            os.replace(temporary, self.champion_path)
        finally:
            temporary.unlink(missing_ok=True)

        history = sorted(self.history_dir.glob("champion_*.pt"))
        for stale in history[:-history_limit]:
            stale.unlink()
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
