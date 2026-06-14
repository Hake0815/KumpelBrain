"""Concrete game execution helpers for seed collection and champion matches."""

from __future__ import annotations

import uuid
from threading import Event

import torch

from game_player import GamePlayer
from inference_service import InferenceClient
from training.rollout_data import RootState
from training.rollout_runner import SeedGamePlayer


def run_seed_game(
    *,
    deck_list1: dict[str, int],
    deck_list2: dict[str, int],
    player1_name: str,
    player2_name: str,
    champion: InferenceClient,
    opponent: InferenceClient,
    compute_device: torch.device,
    seed: int,
    epsilon: float = 0.10,
    temperature: float = 0.5,
) -> list[RootState]:
    done = Event()
    player = SeedGamePlayer(
        deck_list1=deck_list1,
        deck_list2=deck_list2,
        player1_name=player1_name,
        player2_name=player2_name,
        game_uuid=uuid.uuid4(),
        callback_on_game_end=lambda _message: done.set(),
        inference=champion,
        inference_by_player={
            player1_name: champion,
            player2_name: opponent,
        },
        compute_device=compute_device,
        exploration_epsilon=epsilon,
        policy_temperature=temperature,
        seed=seed,
    )
    player.play_game()
    if not done.is_set():
        raise TimeoutError("Seed game did not complete")
    return player.roots


class VersusGamePlayer(GamePlayer):
    def __init__(
        self,
        *args,
        inference_by_player: dict[str, InferenceClient],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.inference_by_player = inference_by_player
        self.winner_name: str | None = None

    def _on_general_update(self, interactions) -> None:
        interaction = interactions[0]
        if interaction.is_game_over():
            self.winner_name = interaction.get_winner_name()
        super()._on_general_update(interactions)

    def _on_player_update(self, interactions, player_name: str) -> None:
        previous_inference = self.inference
        previous_device = self.tensor_device
        self.inference = self.inference_by_player[player_name]
        self.tensor_device = self.inference.tensor_device
        try:
            super()._on_player_update(interactions, player_name)
        finally:
            self.inference = previous_inference
            self.tensor_device = previous_device


def play_candidate_match(
    *,
    deck_list1: dict[str, int],
    deck_list2: dict[str, int],
    candidate: InferenceClient,
    champion: InferenceClient,
    candidate_is_player1: bool,
    compute_device: torch.device,
) -> float:
    player1_name = "player1"
    player2_name = "player2"
    candidate_name = player1_name if candidate_is_player1 else player2_name
    mapping = (
        {player1_name: candidate, player2_name: champion}
        if candidate_is_player1
        else {player1_name: champion, player2_name: candidate}
    )
    done = Event()
    player = VersusGamePlayer(
        deck_list1=deck_list1,
        deck_list2=deck_list2,
        player1_name=player1_name,
        player2_name=player2_name,
        game_uuid=uuid.uuid4(),
        callback_on_game_end=lambda _message: done.set(),
        inference=mapping[player1_name],
        inference_by_player=mapping,
        compute_device=compute_device,
    )
    player.play_game()
    if not done.is_set():
        raise TimeoutError("Champion gate game did not complete")
    if player.winner_name is None:
        return 0.5
    return 1.0 if player.winner_name == candidate_name else 0.0
