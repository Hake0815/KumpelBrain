"""Serialized ProtoBufCardState batches for CardStateEmbedding.forward golden tests."""

from __future__ import annotations

import proto_serialization

from card_embedding_forward_fixtures import apply_card_surface_features


def _pb2_mod():
    return proto_serialization._load_proto_module()


def _ser_card_state(
    card,
    *,
    possible_positions: tuple[int, ...] = (),
    owner: int | None = None,
    opponent_position_knowledge: int | None = None,
) -> bytes:
    pb2 = _pb2_mod()
    st = pb2.ProtoBufCardState()
    st.card.CopyFrom(card)
    pos = st.position
    pos.owner = owner if owner is not None else pb2.OWNER_SELF
    if not possible_positions:
        possible_positions = (pb2.CARD_POSITION_BENCH,)
    for p in possible_positions:
        pos.possible_positions.append(p)
    pos.opponent_position_knowledge = (
        opponent_position_knowledge
        if opponent_position_knowledge is not None
        else pb2.POSITION_KNOWLEDGE_UNKNOWN
    )
    return st.SerializeToString()


def _minimal_pokemon(name: str, deck_id: int):
    pb2 = _pb2_mod()
    c = pb2.ProtoBufCard()
    c.name = name
    c.deck_id = deck_id
    c.card_type = pb2.CARD_TYPE_POKEMON
    c.card_subtype = pb2.CARD_SUBTYPE_BASIC_POKEMON
    c.max_hp = 70
    c.energy_type = pb2.ENERGY_TYPE_WATER
    return c


def _build_fixture_cases() -> dict[str, list[bytes]]:
    cases: dict[str, list[bytes]] = {}
    pb2 = _pb2_mod()

    c0 = _minimal_pokemon("Lonely", 0)
    apply_card_surface_features(c0, 0, 9001)
    cases["single_no_relations"] = [
        _ser_card_state(c0, possible_positions=(pb2.CARD_POSITION_BENCH,))
    ]

    base = _minimal_pokemon("ChainBase", 3)
    mid = _minimal_pokemon("ChainMid", 4)
    top = _minimal_pokemon("ChainTop", 5)
    mid.pre_evolution_ids.append(3)
    top.pre_evolution_ids.append(4)
    for c in (base, mid, top):
        apply_card_surface_features(c, 1, 9002)
    cases["pre_evolution_chain"] = [
        _ser_card_state(base, possible_positions=(pb2.CARD_POSITION_BENCH,)),
        _ser_card_state(mid, possible_positions=(pb2.CARD_POSITION_ACTIVE_SPOT,)),
        _ser_card_state(
            top,
            possible_positions=(pb2.CARD_POSITION_HAND,),
            opponent_position_knowledge=pb2.POSITION_KNOWLEDGE_KNOWN,
        ),
    ]

    parent = _minimal_pokemon("StaticParent", 6)
    child = _minimal_pokemon("StaticChild", 7)
    child.evolves_from = "StaticParent"
    apply_card_surface_features(parent, 2, 9003)
    apply_card_surface_features(child, 3, 9004)
    cases["static_evolves_from"] = [
        _ser_card_state(parent, possible_positions=(pb2.CARD_POSITION_BENCH, pb2.CARD_POSITION_HAND)),
        _ser_card_state(
            child,
            possible_positions=(pb2.CARD_POSITION_ACTIVE_SPOT,),
            owner=pb2.OWNER_OPPONENT,
            opponent_position_knowledge=pb2.POSITION_KNOWLEDGE_NOT_PRIZED,
        ),
    ]

    host = _minimal_pokemon("EnergyHost", 8)
    energy = _minimal_pokemon("EnergyCard", 9)
    host.attached_energy_cards.append(9)
    apply_card_surface_features(host, 4, 9005)
    apply_card_surface_features(energy, 5, 9006)
    cases["attached_energy"] = [
        _ser_card_state(host, possible_positions=(pb2.CARD_POSITION_ATTACHED_TO_CARD,)),
        _ser_card_state(energy, possible_positions=(pb2.CARD_POSITION_FLOATING,)),
    ]

    b = _minimal_pokemon("MixBase", 10)
    e = _minimal_pokemon("MixEnergy", 11)
    s = _minimal_pokemon("MixStage", 12)
    s.pre_evolution_ids.append(10)
    s.attached_energy_cards.append(11)
    s.evolves_from = "MixBase"
    for c in (b, e, s):
        apply_card_surface_features(c, 6, 9007)
    cases["mixed_relations"] = [
        _ser_card_state(b, possible_positions=(pb2.CARD_POSITION_DECK,)),
        _ser_card_state(e, possible_positions=(pb2.CARD_POSITION_PRIZES,)),
        _ser_card_state(s, possible_positions=(pb2.CARD_POSITION_DISCARD_PILE,)),
    ]

    return cases


FIXTURE_CASES: dict[str, list[bytes]] = _build_fixture_cases()

# Must match generator and CardStateEmbedding ctor.
FIXTURE_DIMENSION_OUT = 32


def build_adjacency_divergent_card_bytes() -> list[bytes]:
    """Three serialized ProtoBufCardState rows; card bodies differ pre_evolution vs attached_energy edges."""
    base = _minimal_pokemon("AdjBase", 20)
    energy = _minimal_pokemon("AdjEnergy", 21)
    stage = _minimal_pokemon("AdjStage", 22)
    stage.pre_evolution_ids.append(20)
    stage.attached_energy_cards.append(21)
    for c in (base, energy, stage):
        apply_card_surface_features(c, 7, 9010)
    return [_ser_card_state(c) for c in (base, energy, stage)]
