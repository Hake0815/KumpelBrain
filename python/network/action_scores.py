"""Dual-head action scores used by play and rollout training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionScores:
    value_logits: torch.Tensor
    policy_logits: torch.Tensor

    def masked_fill(self, mask: torch.Tensor, value: float) -> ActionScores:
        return ActionScores(
            self.value_logits.masked_fill(mask, value),
            self.policy_logits.masked_fill(mask, value),
        )

