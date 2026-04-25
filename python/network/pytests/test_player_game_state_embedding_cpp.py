"""C++ PlayerStateEmbedding / GameStateEmbedding / CardPositionEmbedding smoke and uneven-trait tests.

Run: python -m pytest python/network/pytests/test_player_game_state_embedding_cpp.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"
for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import kumpel_embedding  # noqa: E402
import proto_serialization  # noqa: E402


def _pb2():
    return proto_serialization._load_proto_module()


def _player_state_bytes(*, num_traits: int, seed: int = 0) -> bytes:
    pb2 = _pb2()
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


def _game_state_bytes(*, self_traits: int, opp_traits: int, num_card_rows: int = 0) -> bytes:
    pb2 = _pb2()
    g = pb2.ProtoBufGameState()
    g.recreatable = True
    g.technical_game_state = pb2.GAME_STATE_IDLE_PLAYER_TURN
    g.self_state.ParseFromString(_player_state_bytes(num_traits=self_traits, seed=0))
    g.opponent_state.ParseFromString(_player_state_bytes(num_traits=opp_traits, seed=100))
    for i in range(num_card_rows):
        st = g.card_states.add()
        st.card.name = f"Bench{i}"
        st.card.deck_id = i
        st.card.card_type = pb2.CARD_TYPE_POKEMON
        st.card.card_subtype = pb2.CARD_SUBTYPE_BASIC_POKEMON
        st.card.max_hp = 60
        st.card.energy_type = pb2.ENERGY_TYPE_WATER
        pos = st.position
        pos.owner = pb2.OWNER_SELF
        pos.possible_positions.append(pb2.CARD_POSITION_BENCH)
        pos.opponent_position_knowledge = pb2.POSITION_KNOWLEDGE_UNKNOWN
        pos.top_deck_position_index = 0
    return g.SerializeToString()


def test_player_state_embedding_uneven_traits_cpu():
    dim = 32
    device = torch.device("cpu")
    m = kumpel_embedding.PlayerStateEmbedding(dim, device=device, dtype=torch.float32)
    m.eval()
    for a_traits, b_traits in ((0, 4), (1, 3), (4, 0)):
        a = _player_state_bytes(num_traits=a_traits, seed=a_traits)
        b = _player_state_bytes(num_traits=b_traits, seed=b_traits + 10)
        with torch.inference_mode():
            out = m.forward(a, b)
            repeated = m.forward(a, b)
        assert out.shape == (2, dim)
        assert out.device.type == device.type
        assert torch.isfinite(out).all()
        torch.testing.assert_close(repeated, out)


def test_game_state_embedding_zero_cards_uneven_traits_cpu():
    dim = 32
    device = torch.device("cpu")
    m = kumpel_embedding.GameStateEmbedding(dim, device=device, dtype=torch.float32)
    m.eval()
    payload = _game_state_bytes(self_traits=0, opp_traits=4, num_card_rows=0)
    with torch.inference_mode():
        out = m.forward(payload)
    assert out.shape == (2, dim)
    assert torch.isfinite(out).all()


def test_game_state_embedding_with_cards_uneven_traits_cpu():
    dim = 32
    device = torch.device("cpu")
    m = kumpel_embedding.GameStateEmbedding(dim, device=device, dtype=torch.float32)
    m.eval()
    n_cards = 3
    payload = _game_state_bytes(self_traits=1, opp_traits=3, num_card_rows=n_cards)
    with torch.inference_mode():
        out = m.forward(payload)
    assert out.shape == (2 + n_cards, dim)
    assert torch.isfinite(out).all()


def test_card_position_embedding_smoke_cpu():
    import card_state_embedding_forward_fixtures as fixtures  # noqa: E402

    dim = fixtures.FIXTURE_DIMENSION_OUT
    device = torch.device("cpu")
    states = fixtures.FIXTURE_CASES["single_no_relations"]
    shared = kumpel_embedding.SharedEmbeddingHolder(dim, device=device, dtype=torch.float32)
    emb = kumpel_embedding.CardPositionEmbedding(shared, dim, device=device, dtype=torch.float32)
    emb.eval()
    with torch.inference_mode():
        out = emb.forward(states)
    assert out.shape == (len(states), dim)
    assert torch.isfinite(out).all()
