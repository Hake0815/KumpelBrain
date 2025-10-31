from typing import Callable
import csharp_runtime
from gamecore.game import IGameController
from System.Collections.Generic import Dictionary

from interaction_wrapper import InteractionWrapper


class GameControllerWrapper:
    game_controller: IGameController

    def __init__(self, log_file_path: str):
        self.game_controller = IGameController.Create(log_file_path)

    def create_game(
        self,
        deck_list1: dict[str, int],
        deck_list2: dict[str, int],
        player1_name: str,
        player2_name: str,
    ):
        self.game_controller.CreateGame(
            self._convert_deck_list_to_dictionary(deck_list1),
            self._convert_deck_list_to_dictionary(deck_list2),
            player1_name,
            player2_name,
        ).Wait()

    def recreate_game_from_log(self):
        self.game_controller.RecreateGameFromLog()

    def start_game(self):
        self.game_controller.StartGame()

    def get_game_state_for_player(self, player_name: str):
        return self.game_controller.ExportGameState(player_name)

    def subscribe_to_player1_updates(
        self, callback: Callable[[list[InteractionWrapper]], None]
    ):
        self.game_controller.NotifyPlayer1 += self._delegate_callback(callback)

    def subscribe_to_player2_updates(
        self, callback: Callable[[list[InteractionWrapper]], None]
    ):
        self.game_controller.NotifyPlayer2 += self._delegate_callback(callback)

    def subscribe_to_general_updates(
        self, callback: Callable[[list[InteractionWrapper]], None]
    ):
        self.game_controller.NotifyGeneral += self._delegate_callback(callback)

    def _delegate_callback(
        self, callback_wrapper: Callable[[list[InteractionWrapper]], None]
    ):
        return lambda interactions: callback_wrapper(
            [InteractionWrapper(interaction) for interaction in interactions]
        )

    def _convert_deck_list_to_dictionary(self, deck_list: dict[str, int]):
        cs_deck_list = Dictionary[str, int]()
        for key, value in deck_list.items():
            cs_deck_list[key] = value
        return cs_deck_list
