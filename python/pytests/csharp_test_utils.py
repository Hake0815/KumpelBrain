"""Shared helpers for optional C# integration tests."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
GAME_LOGIC_DLL = (
    _REPO_ROOT / "gamecore" / "bin" / "Release" / "net10.0" / "GameLogic.dll"
)
_WRAPPER_DIR = _REPO_ROOT / "python" / "game_logic_wrappers"


@lru_cache(maxsize=1)
def csharp_integration_available() -> bool:
    if os.environ.get("KUMPEL_SKIP_CSHARP_TESTS", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("KUMPEL_ENABLE_CSHARP_TESTS", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return False
    if not GAME_LOGIC_DLL.is_file():
        return False
    probe = (
        "import sys; "
        f"sys.path.insert(0, {str(_WRAPPER_DIR)!r}); "
        "import csharp_runtime; "
        "from gamecore.serialization import ProtoBufGameState"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0
