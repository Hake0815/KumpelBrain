import os
from random import random, randrange, shuffle
from typing import Callable
from game_logic_wrappers.game_controller_wrapper import GameControllerWrapper
from game_logic_wrappers.interaction_wrapper import InteractionWrapper


class GamePlayer:
    game_controller: GameControllerWrapper

    def __init__(
        self,
        log_file_path: str,
        deck_list1: dict[str, int],
        deck_list2: dict[str, int],
        player1_name: str,
        player2_name: str,
        callback_on_game_end: Callable[[str], None],
    ):
        try:
            os.remove(log_file_path)
        except OSError:
            pass

        self.game_controller = GameControllerWrapper(log_file_path)
        self.deck_list1 = deck_list1
        self.deck_list2 = deck_list2
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.callback_on_game_end = callback_on_game_end

    def play_game(self) -> None:
        self.game_controller.subscribe_to_general_updates(self._on_general_update)
        self.game_controller.subscribe_to_player1_updates(self._on_player_update)
        self.game_controller.subscribe_to_player2_updates(self._on_player_update)
        self.game_controller.create_game(
            self.deck_list1, self.deck_list2, self.player1_name, self.player2_name
        )
        self.game_controller.start_game()

    def _on_general_update(self, interactions: list[InteractionWrapper]) -> None:
        interaction = interactions[0]
        if interaction.is_game_over():
            self.callback_on_game_end(interaction.get_game_over_message())
            return

        self._perform_interaction(interaction)

    def _on_player_update(self, interactions: list[InteractionWrapper]) -> None:
        if len(interactions) == 1:
            self._perform_interaction(interactions[0])
        else:
            self._perform_interaction(interactions[randrange(len(interactions))])

    def _perform_interaction(self, interaction: InteractionWrapper) -> None:
        if interaction.is_with_target():
            self._perform_action_with_targets(interaction)
        else:
            interaction.perform_action()

    def _perform_action_with_targets(self, interaction: InteractionWrapper) -> None:
        possible_targets = interaction.get_targets()
        shuffle(possible_targets)
        targets = []
        if interaction.is_with_condition_target():
            i = 0
            while not interaction.is_target_condition_fulfilled(targets):
                targets.append(possible_targets[i])
                i += 1
        else:
            for i in range(interaction.get_number_of_targets()):
                targets.append(possible_targets[i])
        interaction.perform_action_with_targets(targets)
