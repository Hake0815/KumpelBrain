"""Helpers for parsing stored ProtoBufGameState payloads."""

from __future__ import annotations

from proto_serialization import _load_proto_module


def parse_game_state_bytes(state_bytes: bytes):
    """Parse protobuf bytes into a Python ProtoBufGameState message."""
    pb2 = _load_proto_module()
    message = pb2.ProtoBufGameState()
    message.ParseFromString(state_bytes)
    return message


def is_recreatable(state_bytes: bytes) -> bool:
    return parse_game_state_bytes(state_bytes).recreatable


def is_rollout_root(state_bytes: bytes) -> bool:
    state = parse_game_state_bytes(state_bytes)
    enum = state.DESCRIPTOR.fields_by_name["technical_game_state"].enum_type
    technical_state = enum.values_by_number[int(state.technical_game_state)].name
    return bool(state.recreatable) and technical_state == "GAME_STATE_IDLE_PLAYER_TURN"


def player_turn_counters(state_bytes: bytes) -> tuple[int, int]:
    state = parse_game_state_bytes(state_bytes)
    return int(state.self_state.turn_counter), int(state.opponent_state.turn_counter)


def game_state_metadata(state_bytes: bytes) -> tuple[bool, int, int]:
    state = parse_game_state_bytes(state_bytes)
    return (
        bool(state.recreatable),
        int(state.self_state.turn_counter),
        int(state.opponent_state.turn_counter),
    )


def win_value_for_player(player_name: str, winner_name: str | None) -> float:
    if winner_name is None:
        return 0.5
    return 1.0 if player_name == winner_name else 0.0
