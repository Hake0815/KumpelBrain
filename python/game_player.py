import os
from random import random, randrange, shuffle
from typing import Callable
import uuid
from game_logic_wrappers.game_controller_wrapper import GameControllerWrapper
from game_logic_wrappers.interaction_wrapper import InteractionWrapper


class GamePlayer:
    game_controller: GameControllerWrapper
    game_uuid: uuid

    def __init__(
        self,
        deck_list1: dict[str, int],
        deck_list2: dict[str, int],
        player1_name: str,
        player2_name: str,
        game_uuid: uuid,
        callback_on_game_end: Callable[[str], None],
        enable_file_logging: bool = False,
    ):
        self.game_uuid = game_uuid
        self.enable_file_logging = enable_file_logging

        if enable_file_logging:
            try:
                os.remove(f"game_action_logs/log_{game_uuid}.txt")
                os.remove(f"game_state_logs/game_state_{game_uuid}.txt")
            except OSError:
                pass
            log_file_path = f"game_action_logs/log_{game_uuid}.txt"
            self.game_controller = GameControllerWrapper(log_file_path)
            self.game_controller.set_application_log_file_path(
                f"game_application_logs/application_log_{game_uuid}.txt"
            )
        else:
            # Use a minimal log path even when logging is disabled (C# might require it)
            log_file_path = f"game_action_logs/log_{game_uuid}.txt"
            self.game_controller = GameControllerWrapper(log_file_path)

        # Use ERROR level instead of DEBUG for better performance
        self.game_controller.set_application_log_log_level("ERROR")
        self.deck_list1 = deck_list1
        self.deck_list2 = deck_list2
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.callback_on_game_end = callback_on_game_end

    def play_game(self) -> None:
        self.game_controller.subscribe_to_general_updates(self._on_general_update)
        self.game_controller.subscribe_to_player1_updates(self._on_player_1_update)
        self.game_controller.subscribe_to_player2_updates(self._on_player_2_update)
        self.game_controller.create_game(
            self.deck_list1, self.deck_list2, self.player1_name, self.player2_name
        )
        self.game_controller.start_game()

    def _on_general_update(self, interactions: list[InteractionWrapper]) -> None:
        interaction = interactions[0]
        if interaction.is_game_over():
            # interaction.perform_action()
            self.callback_on_game_end(interaction.get_game_over_message())
            return

        self._perform_interaction(interaction)

    def _on_player_1_update(self, interactions: list[InteractionWrapper]) -> None:
        game_state = self.game_controller.export_game_state_as_json_string(
            self.player1_name
        )
        if self.enable_file_logging:
            with open(f"game_state_logs/game_state_{self.game_uuid}.txt", "a") as f:
                f.writelines([game_state, "\n"])
        self._on_player_update(interactions)

    def _on_player_2_update(self, interactions: list[InteractionWrapper]) -> None:
        game_state = self.game_controller.export_game_state_as_json_string(
            self.player2_name
        )
        if self.enable_file_logging:
            with open(f"game_state_logs/game_state_{self.game_uuid}.txt", "a") as f:
                f.write(game_state)
        self._on_player_update(interactions)

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
