"""JSON fixtures for KumpelNetwork smoke tests (ProtoBufGameState / ProtoBufGameInteraction)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from google.protobuf import json_format

import proto_serialization

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GAME_STATE_JSON_PATH = _FIXTURES_DIR / "kumpel_network_smoke_game_state.json"
GAME_INTERACTIONS_JSON_PATH = _FIXTURES_DIR / "kumpel_network_smoke_game_interactions.json"


def _pascal_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _convert_pascal_case_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_pascal_to_snake(k): _convert_pascal_case_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_pascal_case_keys(item) for item in obj]
    return obj


def load_smoke_game_state_json() -> dict[str, Any]:
    return json.loads(GAME_STATE_JSON_PATH.read_text(encoding="utf-8"))


def load_smoke_game_interactions_json() -> list[dict[str, Any]]:
    return json.loads(GAME_INTERACTIONS_JSON_PATH.read_text(encoding="utf-8"))


def game_state_json_to_bytes(game_state_json: dict[str, Any]) -> bytes:
    pb2 = proto_serialization._load_proto_module()
    message = pb2.ProtoBufGameState()
    json_format.Parse(
        json.dumps(_convert_pascal_case_keys(game_state_json)),
        message,
        ignore_unknown_fields=True,
    )
    return message.SerializeToString()


def game_interactions_json_to_bytes(
    interactions_json: list[dict[str, Any]],
) -> list[bytes]:
    pb2 = proto_serialization._load_proto_module()
    out: list[bytes] = []
    for interaction_json in interactions_json:
        message = pb2.ProtoBufGameInteraction()
        json_format.Parse(
            json.dumps(interaction_json),
            message,
            ignore_unknown_fields=True,
        )
        out.append(message.SerializeToString())
    return out


def load_smoke_game_state_bytes() -> bytes:
    return game_state_json_to_bytes(load_smoke_game_state_json())


def load_smoke_game_interaction_bytes() -> list[bytes]:
    return game_interactions_json_to_bytes(load_smoke_game_interactions_json())
