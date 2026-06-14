"""Unit test for parsing game-state bytes through the C# protobuf parser."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from csharp_test_utils import csharp_integration_available

_REPO = Path(__file__).resolve().parents[2]
_WRAPPER = _REPO / "python" / "game_logic_wrappers"
_NETWORK = _REPO / "python" / "network"
_PYTESTS = _NETWORK / "pytests"
for _p in (_WRAPPER, _NETWORK, _PYTESTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

GAME_LOGIC_DLL = (
    _REPO / "gamecore" / "bin" / "Release" / "net10.0" / "GameLogic.dll"
)

pytestmark = pytest.mark.skipif(
    not csharp_integration_available(),
    reason="C# integration disabled; set KUMPEL_ENABLE_CSHARP_TESTS=1 to run",
)


def test_parse_game_state_bytes_roundtrip() -> None:
    import kumpel_network_json_fixtures as fixtures
    from game_controller_wrapper import GameControllerWrapper

    state_bytes = fixtures.load_smoke_game_state_bytes()
    parsed = GameControllerWrapper._parse_game_state_bytes(state_bytes)
    assert parsed.Recreatable in (True, False)
    assert parsed.CalculateSize() > 0
