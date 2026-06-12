"""Tests for process-pool worker resolution."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import main  # noqa: E402


def test_physical_core_count_is_positive() -> None:
    assert main._physical_core_count() >= 1


def test_resolve_workers_defaults_to_physical_cores() -> None:
    physical = main._physical_core_count()
    assert main._resolve_workers(num_games=1000) == physical


def test_resolve_workers_respects_override_and_num_games_cap() -> None:
    import os

    old = os.environ.get("KUMPEL_WORKERS")
    try:
        os.environ["KUMPEL_WORKERS"] = "3"
        assert main._resolve_workers(num_games=2) == 2
        assert main._resolve_workers(num_games=10) == 3
    finally:
        if old is None:
            os.environ.pop("KUMPEL_WORKERS", None)
        else:
            os.environ["KUMPEL_WORKERS"] = old
