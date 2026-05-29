import os
from random import random, randrange, shuffle
from typing import Callable
import uuid

import torch
from game_logic_wrappers.game_controller_wrapper import GameControllerWrapper
from game_logic_wrappers.interaction_wrapper import InteractionWrapper
import json
from network.kumpel_network import KumpelNetwork
from network.multi_head_attention import MultiHeadAttentionArgs
from game_logic_wrappers.card_wrapper import CardWrapper
from network.selector import Selector


class GamePlayer:
    game_controller: GameControllerWrapper
    game_uuid: uuid.UUID

    def __init__(
        self,
        deck_list1: dict[str, int],
        deck_list2: dict[str, int],
        player1_name: str,
        player2_name: str,
        game_uuid: uuid.UUID,
        callback_on_game_end: Callable[[str], None],
        enable_file_logging: bool = False,
    ):
        self.game_uuid = game_uuid
        self.enable_file_logging = enable_file_logging

        log_file_path = f"game_action_logs/log_{game_uuid}.txt"
        self.game_controller = GameControllerWrapper(log_file_path)
        if enable_file_logging:
            try:
                os.remove(f"game_action_logs/log_{game_uuid}.txt")
                os.remove(f"game_state_logs/game_state_{game_uuid}.txt")
            except OSError:
                pass
            self.game_controller.set_application_log_file_path(
                f"game_application_logs/application_log_{game_uuid}.txt"
            )

        self.game_controller.set_application_log_log_level("ERROR")
        self.deck_list1 = deck_list1
        self.deck_list2 = deck_list2
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.callback_on_game_end = callback_on_game_end
        self._initialize_network()

    def _initialize_network(self) -> None:
        dimension_out = 128
        dimension_state_inner = dimension_out * 4
        dimension_interaction_inner = dimension_out * 4
        dimension_target_inner = dimension_out * 4
        num_layers = 12
        compute_device = torch.device("cuda")
        attention_args = MultiHeadAttentionArgs(
            dimension_out,
            dimension_out,
            dimension_out,
            32,
            4,
            bias=False,
            device=compute_device,
        )
        self.network = KumpelNetwork(
            dimension_out,
            dimension_state_inner,
            dimension_interaction_inner,
            attention_args,
            attention_args,
            num_layers,
            device=compute_device,
        )
        self.compute_device = compute_device

        selector_device = torch.device("cpu")
        attention_args_selector = MultiHeadAttentionArgs(
            dimension_out,
            dimension_out,
            dimension_out,
            32,
            4,
            bias=False,
            device=selector_device,
        )
        self.selector = Selector(
            dimension_out,
            dimension_target_inner,
            attention_args_selector,
            device=selector_device,
        )
        self.selector_device = selector_device

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
            self.callback_on_game_end(interaction.get_game_over_message())
            return

        self._perform_interaction(interaction, None, None, None, "")

    def _on_player_1_update(self, interactions: list[InteractionWrapper]) -> None:
        self._log_game_state(self.player1_name)
        self._on_player_update(interactions, self.player1_name)

    def _on_player_2_update(self, interactions: list[InteractionWrapper]) -> None:
        self._log_game_state(self.player2_name)
        self._on_player_update(interactions, self.player2_name)

    def _on_player_update(
        self, interactions: list[InteractionWrapper], player_name: str
    ) -> None:
        if len(interactions) == 1:
            self._perform_interaction(interactions[0], None, None, None, player_name)
        else:
            self._log_interactions(interactions)
            (
                interaction_scores,
                transformed_state,
                embedded_interactions,
                card_indices,
            ) = self._evaluate_game_state(player_name, interactions)
            chosen_interaction_index = int(torch.argmax(interaction_scores).item())
            self._perform_interaction(
                interactions[chosen_interaction_index],
                transformed_state,
                embedded_interactions[chosen_interaction_index],
                card_indices,
                player_name,
            )

    def _evaluate_game_state(
        self, player_name: str, interactions: list[InteractionWrapper]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        game_state = self.game_controller.export_game_state_as_byte_array(player_name)
        return self.network.forward(
            game_state, [interaction.to_bytes() for interaction in interactions]
        )

    def _perform_interaction(
        self,
        interaction: InteractionWrapper,
        transformed_state: torch.Tensor | None,
        embedded_interaction: torch.Tensor | None,
        card_indices: torch.Tensor | None,
        player_name: str,
    ) -> None:
        if interaction.is_with_target():
            self._perform_action_with_targets(
                interaction,
                transformed_state,
                embedded_interaction,
                card_indices,
                player_name,
            )
        else:
            interaction.perform_action()

    def _perform_action_with_targets(
        self,
        interaction: InteractionWrapper,
        transformed_state: torch.Tensor | None,
        embedded_interaction: torch.Tensor | None,
        card_indices: torch.Tensor | None,
        player_name: str,
    ) -> None:
        if (
            transformed_state is None
            or embedded_interaction is None
            or card_indices is None
        ):
            _, transformed_state, embedded_interactions, card_indices = (
                self._evaluate_game_state(player_name, [interaction])
            )
            embedded_interaction = embedded_interactions[0]
        transformed_state = transformed_state.to(self.selector_device)
        embedded_interaction = embedded_interaction.to(self.selector_device)
        card_indices = card_indices.to(self.selector_device)
        if interaction.is_with_condition_target():
            targets = self._select_targets_with_condition(
                interaction, transformed_state, embedded_interaction, card_indices
            )
        else:
            targets = self._select_fixed_number_of_targets(
                interaction, transformed_state, embedded_interaction, card_indices
            )

        interaction.perform_action_with_targets(targets)

    def _select_targets_with_condition(
        self,
        interaction: InteractionWrapper,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> list[CardWrapper]:
        current_selection = []
        while True:
            cadidate_cards = interaction.get_candidates_for_partial_selection(
                current_selection
            )
            if len(cadidate_cards) == 0:
                break
            candidates = torch.tensor(
                self._cards_to_deck_ids(cadidate_cards),
                device=self.selector_device,
                dtype=torch.long,
            )
            current_selection_tensor = torch.tensor(
                self._cards_to_deck_ids(current_selection),
                device=self.selector_device,
                dtype=torch.long,
            )
            condition_fulfilled = interaction.is_target_condition_fulfilled(
                current_selection
            )
            target_scores = self.selector(
                candidates,
                current_selection_tensor,
                transformed_state,
                embedded_interaction,
                card_indices,
                include_stop_token=condition_fulfilled,
            )
            chosen_target_index = int(torch.argmax(target_scores).item())
            if chosen_target_index == len(cadidate_cards):
                break
            current_selection.append(cadidate_cards[chosen_target_index])
        return current_selection

    def _select_fixed_number_of_targets(
        self,
        interaction: InteractionWrapper,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> list[CardWrapper]:
        possible_targets = interaction.get_targets()
        current_selection = []
        for _ in range(interaction.get_number_of_targets()):
            candidates = torch.tensor(
                self._cards_to_deck_ids(possible_targets),
                device=self.selector_device,
                dtype=torch.long,
            )
            current_selection_tensor = torch.tensor(
                self._cards_to_deck_ids(current_selection),
                device=self.selector_device,
                dtype=torch.long,
            )
            target_scores = self.selector(
                candidates,
                current_selection_tensor,
                transformed_state,
                embedded_interaction,
                card_indices,
                include_stop_token=False,
            )
            chosen_target_index = int(torch.argmax(target_scores).item())
            current_selection.append(possible_targets[chosen_target_index])
            if not interaction.is_multi_select():
                possible_targets.remove(current_selection[-1])

        return current_selection

    def _cards_to_deck_ids(self, cards: list[CardWrapper]) -> list[int]:
        return [card.get_deck_id() for card in cards]

    def _log_interactions(self, interactions: list[InteractionWrapper]) -> None:
        if self.enable_file_logging:
            with open(
                f"game_interaction_logs/game_interaction_{self.game_uuid}.txt", "a"
            ) as f:
                f.writelines(
                    [
                        json.dumps(
                            [
                                json.loads(interaction.to_json())
                                for interaction in interactions
                            ]
                        ),
                        "\n",
                    ]
                )

    def _log_game_state(self, player_name: str) -> None:
        if self.enable_file_logging:
            with open(f"game_state_logs/game_state_{self.game_uuid}.txt", "a") as f:
                f.writelines(
                    [
                        self.game_controller.export_game_state_as_json(player_name),
                        "\n",
                    ]
                )
