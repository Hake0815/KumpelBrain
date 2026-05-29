"""Accumulators for self-play inference timing (optional, env-gated)."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch


def profiling_enabled() -> bool:
    return os.environ.get("KUMPEL_PROFILE", "").lower() in ("1", "true", "yes")


@dataclass
class InferenceProfiler:
    """Thread-local-friendly accumulators; one instance per GamePlayer."""

    cs_export_s: float = 0.0
    embed_state_s: float = 0.0
    to_compute_s: float = 0.0
    state_transformer_s: float = 0.0
    cards_to_cpu_s: float = 0.0
    embed_interactions_s: float = 0.0
    interaction_network_s: float = 0.0
    selector_s: float = 0.0
    forward_calls: int = 0
    selector_calls: int = 0

    def sync_device(self, device: torch.device) -> None:
        import torch

        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @contextmanager
    def timed(self, device: torch.device) -> Generator["_TimedSpan", None, None]:
        """Sync before/after and yield a span with `.elapsed` after the block."""
        self.sync_device(device)
        t0 = time.perf_counter()
        span = _TimedSpan()
        try:
            yield span
        finally:
            self.sync_device(device)
            span.elapsed = time.perf_counter() - t0

    def merge(self, other: InferenceProfiler) -> None:
        for name in (
            "cs_export_s",
            "embed_state_s",
            "to_compute_s",
            "state_transformer_s",
            "cards_to_cpu_s",
            "embed_interactions_s",
            "interaction_network_s",
            "selector_s",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.forward_calls += other.forward_calls
        self.selector_calls += other.selector_calls

    def format_report(self) -> str:
        total = (
            self.cs_export_s
            + self.embed_state_s
            + self.to_compute_s
            + self.state_transformer_s
            + self.cards_to_cpu_s
            + self.embed_interactions_s
            + self.interaction_network_s
            + self.selector_s
        )
        if total <= 0:
            return "Profiling: no timed inference (set KUMPEL_PROFILE=1)"

        def pct(x: float) -> str:
            return f"{100.0 * x / total:.1f}%" if total else "0%"

        lines = [
            "Inference profiling (seconds, % of timed total):",
            f"  cs_export:           {self.cs_export_s:8.3f}  {pct(self.cs_export_s)}  (n={self.forward_calls} forwards)",
            f"  embed_state (cpp):   {self.embed_state_s:8.3f}  {pct(self.embed_state_s)}",
            f"  to_compute:          {self.to_compute_s:8.3f}  {pct(self.to_compute_s)}",
            f"  state_transformer:   {self.state_transformer_s:8.3f}  {pct(self.state_transformer_s)}",
            f"  cards_to_cpu:        {self.cards_to_cpu_s:8.3f}  {pct(self.cards_to_cpu_s)}",
            f"  embed_interactions:  {self.embed_interactions_s:8.3f}  {pct(self.embed_interactions_s)}",
            f"  interaction_network: {self.interaction_network_s:8.3f}  {pct(self.interaction_network_s)}",
            f"  selector:            {self.selector_s:8.3f}  {pct(self.selector_s)}  (n={self.selector_calls} calls)",
            f"  timed total:         {total:8.3f}",
        ]
        return "\n".join(lines)


@dataclass
class _TimedSpan:
    elapsed: float = 0.0
