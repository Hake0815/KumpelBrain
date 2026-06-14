"""Tests for protobuf game-state helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_NETWORK = _REPO / "python" / "network"
_PYTESTS = _NETWORK / "pytests"
for _p in (_REPO / "python" / "training", _NETWORK, _PYTESTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import kumpel_network_json_fixtures as fixtures
from training.game_state_serialization import (
    game_state_metadata,
    is_recreatable,
    is_rollout_root,
    parse_game_state_bytes,
    win_value_for_player,
)


def test_parse_game_state_bytes_from_fixture() -> None:
    state_bytes = fixtures.load_smoke_game_state_bytes()
    message = parse_game_state_bytes(state_bytes)
    assert message.recreatable in (True, False)
    assert is_recreatable(state_bytes) == message.recreatable
    assert isinstance(is_rollout_root(state_bytes), bool)
    assert game_state_metadata(state_bytes) == (
        bool(message.recreatable),
        int(message.self_state.turn_counter),
        int(message.opponent_state.turn_counter),
    )


def test_win_value_for_player() -> None:
    assert win_value_for_player("player1", "player1") == 1.0
    assert win_value_for_player("player1", "player2") == 0.0
    assert win_value_for_player("player1", None) == 0.5
