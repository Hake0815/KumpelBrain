"""Invalid ATTACK/ABILITY interaction payload validation for GameEmbedding."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"

for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import game_embedding_fixtures as fixtures  # noqa: E402
import kumpel_embedding  # noqa: E402
import proto_serialization  # noqa: E402

DIM = fixtures.FIXTURE_DIMENSION_OUT


def _three_card_state_and_cards():
    device = torch.device("cpu")
    model = kumpel_embedding.GameEmbedding(DIM, device=device, dtype=torch.float32)
    model.eval()
    game_state = fixtures.EMBED_GAME_STATE_CASES["three_cards"]
    with torch.inference_mode():
        state_emb, _mask, indices = model.embedGameState([game_state])
        cards = fixtures.extract_card_embeddings(state_emb)
    return model, indices, cards


def _interaction_with_attack(attack) -> list[bytes]:
    pb2 = proto_serialization._load_proto_module()
    interaction = pb2.ProtoBufGameInteraction()
    interaction.type = pb2.GAME_INTERACTION_TYPE_SELECT_CARDS
    data = interaction.data.add()
    data.data_type = pb2.GAME_INTERACTION_DATA_TYPE_ATTACK_DATA
    data.attack_data.attack.CopyFrom(attack)
    return [interaction.SerializeToString()]


def _interaction_with_ability(ability) -> list[bytes]:
    pb2 = proto_serialization._load_proto_module()
    interaction = pb2.ProtoBufGameInteraction()
    interaction.type = pb2.GAME_INTERACTION_TYPE_SELECT_CARDS
    data = interaction.data.add()
    data.data_type = pb2.GAME_INTERACTION_DATA_TYPE_ABILITY_DATA
    data.ability_data.ability.CopyFrom(ability)
    return [interaction.SerializeToString()]


def test_embed_game_interaction_rejects_empty_attack() -> None:
    pb2 = proto_serialization._load_proto_module()
    attack = pb2.ProtoBufAttack()
    model, indices, cards = _three_card_state_and_cards()
    with pytest.raises((RuntimeError, ValueError), match="ATTACK_DATA|ProtoBufAttack|instructions|energy"):
        with torch.inference_mode():
            model.embedGameInteraction([_interaction_with_attack(attack)], indices, cards)


def test_embed_game_interaction_rejects_empty_ability() -> None:
    pb2 = proto_serialization._load_proto_module()
    ability = pb2.ProtoBufAbility()
    model, indices, cards = _three_card_state_and_cards()
    with pytest.raises((RuntimeError, ValueError), match="ABILITY_DATA|ProtoBufAbility|instructions"):
        with torch.inference_mode():
            model.embedGameInteraction([_interaction_with_ability(ability)], indices, cards)


def test_embed_game_interaction_rejects_ability_conditions_without_instructions() -> None:
    pb2 = proto_serialization._load_proto_module()
    ability = pb2.ProtoBufAbility()
    ability.conditions.add().condition_type = pb2.CONDITION_TYPE_ABILITY_NOT_USED
    model, indices, cards = _three_card_state_and_cards()
    with pytest.raises((RuntimeError, ValueError), match="ABILITY_DATA|conditions without instructions"):
        with torch.inference_mode():
            model.embedGameInteraction([_interaction_with_ability(ability)], indices, cards)
