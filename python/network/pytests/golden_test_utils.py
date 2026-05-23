"""Shared helpers for embedding golden tests."""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch

GOLDEN_SEED = 42
STRICT_RTOL = 0.0
ATOL = 1e-5
# Default for non-golden tests that may tolerate minor numeric drift.
RTOL = 1e-4

_PYTESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = _PYTESTS_DIR / "fixtures"


@contextlib.contextmanager
def deterministic_algorithms(enabled: bool):
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def seed_for_device(device: torch.device) -> None:
    torch.manual_seed(GOLDEN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(GOLDEN_SEED)


def load_golden(filename: str) -> dict[str, torch.Tensor]:
    path = FIXTURES_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)
