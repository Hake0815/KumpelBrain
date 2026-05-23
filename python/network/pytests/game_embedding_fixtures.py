"""Serialized ProtoBufGameState / ProtoBufGameInteraction fixtures for GameEmbedding golden tests."""

from __future__ import annotations

import proto_serialization
import torch

from card_embedding_forward_fixtures import (
    build_expected_card_indices,
    make_card_empty_globals_attack_only,
    make_card_for_variant,
)
from card_state_embedding_forward_fixtures import FIXTURE_CASES as CARD_STATE_FIXTURE_CASES

FIXTURE_DIMENSION_OUT = 32

# Deck ids used by golden interaction fixtures after _game_state_bytes rewrites card rows to 0..n-1.
GOLDEN_THREE_CARD_DECK_IDS = (0, 1, 2)

BENCHMARKED_GAME_INTERACTION_DATA_TYPES = (
    "GAME_INTERACTION_DATA_TYPE_NUMBER_DATA",
    "GAME_INTERACTION_DATA_TYPE_TARGET_DATA",
    "GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA",
    "GAME_INTERACTION_DATA_TYPE_ATTACK_DATA",
    "GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA",
    "GAME_INTERACTION_DATA_TYPE_ABILITY_DATA",
)


def _pb2_mod():
    return proto_serialization._load_proto_module()


def _player_state_bytes(*, num_traits: int, seed: int = 0) -> bytes:
    pb2 = _pb2_mod()
    p = pb2.ProtoBufPlayerState()
    p.is_active = True
    p.is_attacking = False
    p.knows_his_prizes = True
    p.hand_count = 3
    p.deck_count = 40
    p.prizes_count = 3
    p.bench_count = 2
    p.discard_pile_count = 5
    p.turn_counter = 2
    for i in range(num_traits):
        p.player_turn_traits.append((seed + i) % 4)
    return p.SerializeToString()


def _game_state_bytes(
    card_state_rows: list[bytes],
    *,
    self_traits: int = 2,
    opp_traits: int = 2,
    rewrite_deck_ids: bool = True,
) -> bytes:
    pb2 = _pb2_mod()
    g = pb2.ProtoBufGameState()
    g.recreatable = True
    g.technical_game_state = pb2.GAME_STATE_IDLE_PLAYER_TURN
    g.self_state.ParseFromString(_player_state_bytes(num_traits=self_traits, seed=0))
    g.opponent_state.ParseFromString(_player_state_bytes(num_traits=opp_traits, seed=100))
    for row in card_state_rows:
        g.card_states.add().ParseFromString(row)
    if rewrite_deck_ids:
        for i in range(len(g.card_states)):
            g.card_states[i].card.deck_id = i
    return g.SerializeToString()


def build_expected_game_state_card_indices(
    game_state_bytes: bytes, device: torch.device
) -> torch.Tensor:
    """Build deck-id lookup matching C++ CardEmbedding / embedGameState card_indices."""
    pb2 = _pb2_mod()
    game_state = pb2.ProtoBufGameState()
    game_state.ParseFromString(game_state_bytes)
    rows = [card_state.SerializeToString() for card_state in game_state.card_states]
    return build_expected_card_indices(rows, device)


def extract_card_embeddings(game_state_embedding: torch.Tensor) -> torch.Tensor:
    return game_state_embedding[2:]


def _conditional_target_query_leaf(pb2, *, min_targets: int, max_targets: int):
    query = pb2.ProtoBufConditionalTargetQuery()
    query.int_range.min = min_targets
    query.int_range.max = max_targets
    query.selection_qualifier = pb2.SELECTION_QUALIFIER_NUMBER_OF_CARDS
    return query


def _seed_offset_for_type(data_type_name: str) -> int:
    offsets = {
        "GAME_INTERACTION_DATA_TYPE_NUMBER_DATA": 0,
        "GAME_INTERACTION_DATA_TYPE_TARGET_DATA": 100,
        "GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA": 200,
        "GAME_INTERACTION_DATA_TYPE_ATTACK_DATA": 300,
        "GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA": 400,
        "GAME_INTERACTION_DATA_TYPE_ABILITY_DATA": 500,
    }
    return offsets.get(data_type_name, 0)


def _make_game_interaction_data_for_type(
    data_type_name: str,
    seed: int,
    *,
    target_deck_ids: tuple[int, ...] = GOLDEN_THREE_CARD_DECK_IDS,
    use_conditional_target_query: bool = False,
):
    pb2 = _pb2_mod()
    data = pb2.ProtoBufGameInteractionData()
    data.data_type = getattr(pb2, data_type_name)
    if data_type_name == "GAME_INTERACTION_DATA_TYPE_NUMBER_DATA":
        data.number_data.number = 1 + int(seed % 5)
    elif data_type_name == "GAME_INTERACTION_DATA_TYPE_TARGET_DATA":
        target = data.target_data
        target.possible_targets.extend(target_deck_ids)
        target.target_action = pb2.ACTION_ON_SELECTION_DISCARD
        target.remainder_action = pb2.ACTION_ON_SELECTION_TAKE_TO_HAND
        target.allow_multiple_times = seed % 2 == 0
        if use_conditional_target_query:
            target.conditional_target_query.CopyFrom(
                _conditional_target_query_leaf(pb2, min_targets=1, max_targets=2)
            )
        else:
            target.number_of_targets = 2
    elif data_type_name == "GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA":
        data.interaction_card_data.card = target_deck_ids[int(seed % len(target_deck_ids))]
    elif data_type_name == "GAME_INTERACTION_DATA_TYPE_ATTACK_DATA":
        attack_card = make_card_empty_globals_attack_only()
        data.attack_data.attack.CopyFrom(attack_card.attacks[0])
    elif data_type_name == "GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA":
        data.select_from_data.select_from = (
            pb2.SELECT_FROM_DISCARD_PILE if seed % 2 == 0 else pb2.SELECT_FROM_DECK
        )
    elif data_type_name == "GAME_INTERACTION_DATA_TYPE_ABILITY_DATA":
        ability_card = make_card_for_variant(3, seed)
        data.ability_data.ability.CopyFrom(ability_card.ability)
    return data


def make_interaction_bytes(
    batch_size: int,
    *,
    interaction_type_name: str = "GAME_INTERACTION_TYPE_SELECT_CARDS",
    excluded_type: str | None = None,
    target_deck_ids: tuple[int, ...] = GOLDEN_THREE_CARD_DECK_IDS,
    use_conditional_target_query: bool = False,
    data_types: tuple[str, ...] = BENCHMARKED_GAME_INTERACTION_DATA_TYPES,
) -> list[bytes]:
    pb2 = _pb2_mod()
    batch: list[bytes] = []
    for i in range(batch_size):
        interaction = pb2.ProtoBufGameInteraction()
        interaction.type = getattr(pb2, interaction_type_name)
        for data_type_name in data_types:
            if excluded_type is not None and data_type_name == excluded_type:
                continue
            interaction.data.add().CopyFrom(
                _make_game_interaction_data_for_type(
                    data_type_name,
                    i + _seed_offset_for_type(data_type_name),
                    target_deck_ids=target_deck_ids,
                    use_conditional_target_query=use_conditional_target_query,
                )
            )
        batch.append(interaction.SerializeToString())
    return batch


def _build_embed_game_state_cases() -> dict[str, bytes]:
    cases: dict[str, bytes] = {}
    cases["empty"] = _game_state_bytes([], self_traits=2, opp_traits=2)
    cases["three_cards"] = _game_state_bytes(
        CARD_STATE_FIXTURE_CASES["pre_evolution_chain"][:3],
        self_traits=2,
        opp_traits=2,
    )
    cases["uneven_traits"] = _game_state_bytes([], self_traits=0, opp_traits=4)
    cases["mixed_relations"] = _game_state_bytes(
        CARD_STATE_FIXTURE_CASES["mixed_relations"],
        self_traits=2,
        opp_traits=2,
    )
    return cases


def _three_card_game_state_bytes() -> bytes:
    return _game_state_bytes(
        CARD_STATE_FIXTURE_CASES["pre_evolution_chain"][:3],
        self_traits=2,
        opp_traits=2,
    )


def _build_embed_game_interaction_cases() -> dict[str, tuple[bytes, list[bytes]]]:
    empty_state = _game_state_bytes([], self_traits=2, opp_traits=2)
    three_card_state = _three_card_game_state_bytes()
    return {
        "empty": (empty_state, []),
        "all_types_one": (three_card_state, make_interaction_bytes(1)),
        "all_types_four": (three_card_state, make_interaction_bytes(4)),
        "without_TARGET_DATA": (
            three_card_state,
            make_interaction_bytes(2, excluded_type="GAME_INTERACTION_DATA_TYPE_TARGET_DATA"),
        ),
        "without_ABILITY_DATA": (
            three_card_state,
            make_interaction_bytes(2, excluded_type="GAME_INTERACTION_DATA_TYPE_ABILITY_DATA"),
        ),
    }


# Golden tests only cover these case ids (see fixtures/*.pt).
EMBED_GAME_STATE_CASES: dict[str, bytes] = _build_embed_game_state_cases()
EMBED_GAME_INTERACTION_CASES: dict[str, tuple[bytes, list[bytes]]] = (
    _build_embed_game_interaction_cases()
)

# Additional scenarios for non-golden tests (native deck ids, extra interaction types).
EXTENDED_GAME_STATE_CASES: dict[str, bytes] = {
    "sparse_deck_ids": _game_state_bytes(
        CARD_STATE_FIXTURE_CASES["mixed_relations"],
        self_traits=2,
        opp_traits=2,
        rewrite_deck_ids=False,
    ),
}

EXTENDED_GAME_INTERACTION_CASES: dict[str, tuple[bytes, list[bytes]]] = {
    "conditional_target_query": (
        _three_card_game_state_bytes(),
        make_interaction_bytes(
            1,
            excluded_type="GAME_INTERACTION_DATA_TYPE_NUMBER_DATA",
            use_conditional_target_query=True,
        ),
    ),
    "sparse_deck_ids": (
        EXTENDED_GAME_STATE_CASES["sparse_deck_ids"],
        make_interaction_bytes(1, target_deck_ids=(10, 11, 12)),
    ),
    "play_card": (
        _three_card_game_state_bytes(),
        make_interaction_bytes(
            1,
            interaction_type_name="GAME_INTERACTION_TYPE_PLAY_CARD",
            data_types=("GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA",),
        ),
    ),
    "perform_attack": (
        _three_card_game_state_bytes(),
        make_interaction_bytes(
            1,
            interaction_type_name="GAME_INTERACTION_TYPE_PERFORM_ATTACK",
            data_types=(
                "GAME_INTERACTION_DATA_TYPE_ATTACK_DATA",
                "GAME_INTERACTION_DATA_TYPE_TARGET_DATA",
            ),
        ),
    ),
    "retreat": (
        _three_card_game_state_bytes(),
        make_interaction_bytes(
            1,
            interaction_type_name="GAME_INTERACTION_TYPE_RETREAT",
            data_types=("GAME_INTERACTION_DATA_TYPE_NUMBER_DATA",),
        ),
    ),
    "end_turn": (
        _three_card_game_state_bytes(),
        make_interaction_bytes(
            1,
            interaction_type_name="GAME_INTERACTION_TYPE_END_TURN",
            data_types=(),
        ),
    ),
    "mulligan": (
        _game_state_bytes([], self_traits=2, opp_traits=2),
        make_interaction_bytes(
            1,
            interaction_type_name="GAME_INTERACTION_TYPE_SELECT_MULLIGANS",
            data_types=("GAME_INTERACTION_DATA_TYPE_MULLIGAN_DATA",),
        ),
    ),
    "game_over": (
        _game_state_bytes([], self_traits=2, opp_traits=2),
        make_interaction_bytes(
            1,
            interaction_type_name="GAME_INTERACTION_TYPE_GAME_OVER",
            data_types=("GAME_INTERACTION_DATA_TYPE_WINNER_DATA",),
        ),
    ),
}
