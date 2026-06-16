"""Opt-in C# integration test for forced rollout recreation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

from csharp_test_utils import csharp_integration_available

_REPO = Path(__file__).resolve().parents[2]
for path in (
    _REPO / "cpp" / "build",
    _REPO / "python",
    _REPO / "python" / "network",
    _REPO / "python" / "network" / "pytests",
    _REPO / "python" / "training",
    _REPO / "python" / "game_logic_wrappers",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytestmark = pytest.mark.skipif(
    not csharp_integration_available(),
    reason="C# integration disabled; set KUMPEL_ENABLE_CSHARP_TESTS=1 to run",
)


def test_seed_root_can_force_an_interaction_and_continue() -> None:
    from inference_models import create_self_play_models
    from inference_service import DirectInferenceClient
    from main import create_deck_list
    from training.gameplay import run_seed_game
    from training.rollout_runner import ContinuationRolloutRunner

    network, selector, device = create_self_play_models(torch.device("cpu"))
    inference = DirectInferenceClient(network, selector, device)
    roots = run_seed_game(
        deck_list1=create_deck_list(),
        deck_list2=create_deck_list(),
        player1_name="player1",
        player2_name="player2",
        champion=inference,
        opponent=inference,
        compute_device=device,
        seed=0,
    )
    if not roots:
        pytest.skip("Seed game produced no recreatable trainable roots")

    root = roots[len(roots) // 2]
    result = ContinuationRolloutRunner(
        inference,
        lambda: inference,
        device,
        seed=1,
    ).run(root, 0)

    assert result.success, result.mismatch
    assert result.score in {0.0, 0.5, 1.0}


def test_recreation_does_not_skip_card_after_attached_card() -> None:
    from gamecore.serialization import (
        ProtoBufCardPosition,
        ProtoBufOwner,
        ProtoBufTechnicalGameState,
    )
    from game_logic_wrappers.game_controller_wrapper import GameControllerWrapper
    from main import create_deck_list

    with tempfile.NamedTemporaryFile() as log_file:
        source = GameControllerWrapper(log_file.name)
        source.create_game(
            create_deck_list(),
            create_deck_list(),
            "player1",
            "player2",
        )
        state = source.export_game_state("player1")

    state.TechnicalGameState = ProtoBufTechnicalGameState.GameStateIdlePlayerTurn
    state.SelfState.IsActive = True
    state.OpponentState.IsActive = False
    state.SelfState.HandCount = 0
    state.OpponentState.HandCount = 0
    state.SelfState.PrizesCount = 0
    state.OpponentState.PrizesCount = 0

    for card_state in state.CardStates:
        card_state.Position.Owner = (
            ProtoBufOwner.OwnerSelf
            if card_state.Card.DeckId < 60
            else ProtoBufOwner.OwnerOpponent
        )
        card_state.Position.PossiblePositions.Clear()
        card_state.Position.PossiblePositions.Add(
            ProtoBufCardPosition.CardPositionDeck
        )

    # Deck ID 15 is Dreepy and 16 is Drakloak in the standard deck.
    attached = state.CardStates[15]
    attached.Position.PossiblePositions.Clear()
    attached.Position.PossiblePositions.Add(
        ProtoBufCardPosition.CardPositionAttachedToCard
    )
    attached.Position.AttachedToPokemonId = 16

    player1_active = state.CardStates[16]
    player1_active.Position.PossiblePositions.Clear()
    player1_active.Position.PossiblePositions.Add(
        ProtoBufCardPosition.CardPositionActiveSpot
    )

    player2_active = state.CardStates[68]
    player2_active.Position.PossiblePositions.Clear()
    player2_active.Position.PossiblePositions.Add(
        ProtoBufCardPosition.CardPositionActiveSpot
    )

    with tempfile.NamedTemporaryFile() as log_file:
        recreated = GameControllerWrapper(log_file.name)
        recreated.recreate_game_from_game_state(
            state,
            create_deck_list(),
            create_deck_list(),
            "player1",
            "player2",
        )

        assert recreated.game_controller.Game.Player1.ActivePokemon.DeckId == 16
        assert recreated.game_controller.Game.Player2.ActivePokemon.DeckId == 68
