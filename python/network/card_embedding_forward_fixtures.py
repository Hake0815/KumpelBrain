"""Serialized ProtoBufCard batches mirroring cpp/network/benchmark_main.cpp card builders.

Used by golden tests for kumpel_embedding.CardEmbedding.forward.
"""

from __future__ import annotations

import proto_serialization


def _pb2_mod():
    return proto_serialization._load_proto_module()


def _make_leaf_filter(field: int, operation: int, value: int):
    pb2 = _pb2_mod()
    f = pb2.ProtoBufFilter()
    f.is_leaf = True
    f.logical_operator = pb2.FILTER_LOGICAL_OPERATOR_NONE
    f.condition.field = field
    f.condition.operation = operation
    f.condition.value = value
    return f


def _make_group_filter(logical_operator: int, operands: list):
    pb2 = _pb2_mod()
    f = pb2.ProtoBufFilter()
    f.is_leaf = False
    f.logical_operator = logical_operator
    for o in operands:
        f.operands.add().CopyFrom(o)
    return f


def _make_nested_filter(batch_index: int, instruction_index: int):
    pb2 = _pb2_mod()
    card_type = (batch_index + instruction_index) % 4
    card_subtype = 1 + ((batch_index + instruction_index) % 8)
    hp_threshold = 40 + ((batch_index * 7 + instruction_index * 11) % 220)
    return _make_group_filter(
        pb2.FILTER_LOGICAL_OPERATOR_OR,
        [
            _make_group_filter(
                pb2.FILTER_LOGICAL_OPERATOR_AND,
                [
                    _make_leaf_filter(
                        pb2.FILTER_TYPE_CARD_TYPE,
                        pb2.FILTER_OPERATION_EQUALS,
                        card_type,
                    ),
                    _make_leaf_filter(
                        pb2.FILTER_TYPE_HP,
                        pb2.FILTER_OPERATION_GREATER_THAN_OR_EQUAL,
                        hp_threshold,
                    ),
                ],
            ),
            _make_leaf_filter(
                pb2.FILTER_TYPE_CARD_SUBTYPE,
                pb2.FILTER_OPERATION_EQUALS,
                card_subtype,
            ),
        ],
    )


def _make_attack_data(damage: int):
    pb2 = _pb2_mod()
    data = pb2.ProtoBufInstructionData()
    data.instruction_data_type = pb2.INSTRUCTION_DATA_TYPE_ATTACK_DATA
    data.attack_data.attack_target = pb2.ATTACK_TARGET_DEFENDING_POKEMON
    data.attack_data.damage = damage
    return data


def _make_discard_data(source: int):
    pb2 = _pb2_mod()
    data = pb2.ProtoBufInstructionData()
    data.instruction_data_type = pb2.INSTRUCTION_DATA_TYPE_DISCARD_DATA
    data.discard_data.target_source = source
    return data


def _make_card_amount_data(min_amount: int, max_amount: int, from_position: int):
    pb2 = _pb2_mod()
    data = pb2.ProtoBufInstructionData()
    data.instruction_data_type = pb2.INSTRUCTION_DATA_TYPE_CARD_AMOUNT_DATA
    data.card_amount_data.amount.min = min_amount
    data.card_amount_data.amount.max = max_amount
    data.card_amount_data.from_position = from_position
    return data


def _make_return_to_deck_type_data(return_type: int, from_position: int):
    pb2 = _pb2_mod()
    data = pb2.ProtoBufInstructionData()
    data.instruction_data_type = pb2.INSTRUCTION_DATA_TYPE_RETURN_TO_DECK_TYPE_DATA
    data.return_to_deck_type_data.return_to_deck_type = return_type
    data.return_to_deck_type_data.from_position = from_position
    return data


def _make_filter_data(filter_msg):
    pb2 = _pb2_mod()
    data = pb2.ProtoBufInstructionData()
    data.instruction_data_type = pb2.INSTRUCTION_DATA_TYPE_FILTER_DATA
    data.filter_data.filter.CopyFrom(filter_msg)
    return data


def _make_player_target_data(target: int):
    pb2 = _pb2_mod()
    data = pb2.ProtoBufInstructionData()
    data.instruction_data_type = pb2.INSTRUCTION_DATA_TYPE_PLAYER_TARGET_DATA
    data.player_target_data.player_target = target
    return data


def _make_instruction(instruction_type: int, data_entries: list):
    pb2 = _pb2_mod()
    inst = pb2.ProtoBufInstruction()
    inst.instruction_type = instruction_type
    for entry in data_entries:
        inst.data.add().CopyFrom(entry)
    return inst


def _make_condition(condition_type: int, data_entries: list):
    pb2 = _pb2_mod()
    cond = pb2.ProtoBufCondition()
    cond.condition_type = condition_type
    for entry in data_entries:
        cond.data.add().CopyFrom(entry)
    return cond


def make_card_for_variant(variant: int, seed: int):
    """Match benchmark_main.cpp make_card_for_variant(variant % 12, seed)."""
    pb2 = _pb2_mod()
    v = variant % 12
    card = pb2.ProtoBufCard()
    if v == 0:
        card.instructions.add().CopyFrom(
            _make_instruction(pb2.INSTRUCTION_TYPE_SHOW_CARDS, [])
        )
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
    elif v == 1:
        card.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DEAL_DAMAGE,
                [_make_attack_data(7 + int(seed % 50))],
            )
        )
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
    elif v == 2:
        card.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DEAL_DAMAGE,
                [_make_attack_data(10 + int(seed % 40))],
            )
        )
        card.conditions.add().CopyFrom(
            _make_condition(
                pb2.CONDITION_TYPE_HAS_CARDS,
                [
                    _make_card_amount_data(1, 4, pb2.CARD_POSITION_DECK),
                    _make_filter_data(_make_nested_filter(seed, 0)),
                ],
            )
        )
    elif v == 3:
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
        ab = card.ability
        ab.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DEAL_DAMAGE,
                [_make_attack_data(15 + int(seed % 30))],
            )
        )
    elif v == 4:
        ab = card.ability
        ab.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
        ab.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DISCARD, [_make_discard_data(0)]
            )
        )
        atk = card.attacks.add()
        atk.energy_cost.append(pb2.ENERGY_TYPE_GRASS)
        atk.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DEAL_DAMAGE,
                [_make_attack_data(20 + int(seed % 25))],
            )
        )
    elif v == 5:
        card.instructions.add().CopyFrom(
            _make_instruction(pb2.INSTRUCTION_TYPE_SHOW_CARDS, [])
        )
        card.conditions.add().CopyFrom(
            _make_condition(
                pb2.CONDITION_TYPE_HAS_CARDS,
                [
                    _make_card_amount_data(1, 8, pb2.CARD_POSITION_HAND),
                    _make_filter_data(
                        _make_leaf_filter(
                            pb2.FILTER_TYPE_CARD_TYPE,
                            pb2.FILTER_OPERATION_EQUALS,
                            1,
                        )
                    ),
                ],
            )
        )
    elif v == 6:
        ab = card.ability
        ab.conditions.add().CopyFrom(
            _make_condition(
                pb2.CONDITION_TYPE_HAS_CARDS,
                [
                    _make_card_amount_data(1, 3, pb2.CARD_POSITION_DECK),
                    _make_filter_data(_make_nested_filter(seed, 1)),
                ],
            )
        )
        ab.instructions.add().CopyFrom(
            _make_instruction(pb2.INSTRUCTION_TYPE_SHOW_CARDS, [])
        )
    elif v == 7:
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
        for a in range(2):
            atk = card.attacks.add()
            atk.energy_cost.append((seed + a) % 3 + 1)
            atk.instructions.add().CopyFrom(
                _make_instruction(
                    pb2.INSTRUCTION_TYPE_DEAL_DAMAGE,
                    [_make_attack_data(12 + a + int(seed % 20))],
                )
            )
    elif v == 8:
        card.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_SELECT_CARDS,
                [
                    _make_card_amount_data(1, 2, pb2.CARD_POSITION_HAND),
                    _make_filter_data(_make_nested_filter(seed, 2)),
                ],
            )
        )
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
        ab = card.ability
        ab.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_SHUFFLE_DECK,
                [_make_player_target_data(int(seed % 2))],
            )
        )
    elif v == 9:
        card.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_PUT_IN_DECK,
                [
                    _make_return_to_deck_type_data(
                        int(seed % 2), pb2.CARD_POSITION_DISCARD_PILE
                    )
                ],
            )
        )
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
        ab = card.ability
        ab.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
        ab.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_REVEAL_CARDS,
                [
                    _make_card_amount_data(
                        1, 2, pb2.CARD_POSITION_SELECTED_CARDS
                    ),
                    _make_filter_data(
                        _make_leaf_filter(
                            pb2.FILTER_TYPE_EXCLUDE_SOURCE,
                            pb2.FILTER_OPERATION_NONE,
                            0,
                        )
                    ),
                ],
            )
        )
    elif v == 10:
        card.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DEAL_DAMAGE,
                [_make_attack_data(5 + int(seed % 50))],
            )
        )
        card.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DISCARD,
                [_make_discard_data(int(seed % 3))],
            )
        )
        card.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_TAKE_TO_HAND,
                [_make_card_amount_data(1, 1, pb2.CARD_POSITION_DECK)],
            )
        )
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
    elif v == 11:
        card.conditions.add().CopyFrom(
            _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
        )
        atk = card.attacks.add()
        atk.instructions.add().CopyFrom(
            _make_instruction(
                pb2.INSTRUCTION_TYPE_DEAL_DAMAGE,
                [_make_attack_data(30 + int(seed % 40))],
            )
        )
    return card


def build_card_batch(batch_size: int) -> list:
    return [
        make_card_for_variant(int(i % 12), i) for i in range(batch_size)
    ]


def make_card_empty_global_instructions_one_condition():
    pb2 = _pb2_mod()
    card = pb2.ProtoBufCard()
    card.conditions.add().CopyFrom(
        _make_condition(pb2.CONDITION_TYPE_ABILITY_NOT_USED, [])
    )
    return card


def make_card_empty_global_conditions_one_instruction():
    pb2 = _pb2_mod()
    card = pb2.ProtoBufCard()
    card.instructions.add().CopyFrom(
        _make_instruction(pb2.INSTRUCTION_TYPE_SHOW_CARDS, [])
    )
    return card


def make_card_empty_globals_attack_only():
    pb2 = _pb2_mod()
    card = pb2.ProtoBufCard()
    atk = card.attacks.add()
    atk.energy_cost.append(pb2.ENERGY_TYPE_GRASS)
    atk.instructions.add().CopyFrom(
        _make_instruction(
            pb2.INSTRUCTION_TYPE_DEAL_DAMAGE, [_make_attack_data(42)]
        )
    )
    return card


def make_card_completely_empty():
    return _pb2_mod().ProtoBufCard()


def _ser(card) -> bytes:
    return card.SerializeToString()


def build_fixture_cases() -> dict[str, list[bytes]]:
    """Stable case ids -> list of serialized ProtoBufCard (order matches batch)."""
    cases: dict[str, list[bytes]] = {}

    for v in range(12):
        cases[f"single_variant_{v}"] = [_ser(make_card_for_variant(v, v))]

    cases["batch_twelve_variants"] = [
        _ser(make_card_for_variant(i, i)) for i in range(12)
    ]
    cases["batch_small_mixed"] = [_ser(c) for c in build_card_batch(4)]

    cases["completely_empty"] = [_ser(make_card_completely_empty())]
    cases["empty_global_instructions_one_condition"] = [
        _ser(make_card_empty_global_instructions_one_condition())
    ]
    cases["empty_global_conditions_one_instruction"] = [
        _ser(make_card_empty_global_conditions_one_instruction())
    ]
    cases["empty_globals_attack_only"] = [
        _ser(make_card_empty_globals_attack_only())
    ]
    cases["batch_mixed_empty_global_and_empty_card"] = [
        _ser(make_card_empty_global_instructions_one_condition()),
        _ser(make_card_empty_global_conditions_one_instruction()),
        _ser(make_card_empty_globals_attack_only()),
        _ser(make_card_completely_empty()),
    ]
    # Empty card batch is not supported: MultiHeadAttention broadcast fails for batch_size==0.

    return cases


FIXTURE_CASES: dict[str, list[bytes]] = build_fixture_cases()

# Golden tests use this dimension (must match generator and CardEmbedding ctor).
FIXTURE_DIMENSION_OUT = 32
