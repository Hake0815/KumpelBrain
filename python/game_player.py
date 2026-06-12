import os
import time
from typing import Callable
import uuid

import torch
from game_logic_wrappers.game_controller_wrapper import GameControllerWrapper
from game_logic_wrappers.interaction_wrapper import InteractionWrapper
import json
from network.kumpel_network import KumpelNetwork
from network.selector import Selector
from game_logic_wrappers.card_wrapper import CardWrapper
from profiling import InferenceProfiler, profiling_enabled


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
        network: KumpelNetwork,
        selector: Selector,
        compute_device: torch.device,
        enable_file_logging: bool = False,
        enable_profiling: bool | None = None,
    ):
        self.game_uuid = game_uuid
        self.enable_file_logging = enable_file_logging
        self.network = network
        self.selector = selector
        self.compute_device = compute_device
        self.selector_device = compute_device

        if enable_profiling is None:
            enable_profiling = profiling_enabled()
        self.profiler: InferenceProfiler | None = (
            InferenceProfiler() if enable_profiling else None
        )
        self.network.profiler = self.profiler

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
            if self.profiler is not None:
                print(self.profiler.format_report())
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
            with torch.inference_mode():
                (
                    interaction_scores,
                    transformed_state,
                    embedded_interactions,
                    card_indices,
                ) = self._evaluate_game_state(player_name, interactions)
                chosen_interaction_index = int(interaction_scores.argmax().item())
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
        if self.profiler is not None:
            self.profiler.sync_device(self.compute_device)
            t0 = time.perf_counter()
            game_state = self.game_controller.export_game_state_as_byte_array(
                player_name
            )
            self.profiler.sync_device(self.compute_device)
            self.profiler.cs_export_s += time.perf_counter() - t0
            return self.network.forward(
                game_state, [interaction.to_bytes() for interaction in interactions]
            )

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
        with torch.inference_mode():
            if (
                transformed_state is None
                or embedded_interaction is None
                or card_indices is None
            ):
                _, transformed_state, embedded_interactions, card_indices = (
                    self._evaluate_game_state(player_name, [interaction])
                )
                embedded_interaction = embedded_interactions[0]

            if transformed_state.device != self.selector_device:
                transformed_state = transformed_state.to(self.selector_device)
                embedded_interaction = embedded_interaction.to(self.selector_device)
                card_indices = card_indices.to(self.selector_device)

            if interaction.is_with_condition_target():
                targets = self._select_targets_with_condition(
                    interaction,
                    transformed_state,
                    embedded_interaction,
                    card_indices,
                )
            else:
                targets = self._select_fixed_number_of_targets(
                    interaction,
                    transformed_state,
                    embedded_interaction,
                    card_indices,
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
            current_selection_tensor = self._deck_ids_tensor(
                self._cards_to_deck_ids(current_selection)
            )
            condition_fulfilled = interaction.is_target_condition_fulfilled(
                current_selection
            )
            target_scores = self._score_targets(
                candidates,
                current_selection_tensor,
                transformed_state,
                embedded_interaction,
                card_indices,
                include_stop_token=condition_fulfilled,
            )
            chosen_target_index = int(target_scores.argmax().item())
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
        candidates = self._deck_ids_tensor(self._cards_to_deck_ids(possible_targets))
        empty_selection = self._deck_ids_tensor([])
        for _ in range(interaction.get_number_of_targets()):
            current_selection_tensor = (
                empty_selection
                if not current_selection
                else self._deck_ids_tensor(self._cards_to_deck_ids(current_selection))
            )
            target_scores = self._score_targets(
                candidates,
                current_selection_tensor,
                transformed_state,
                embedded_interaction,
                card_indices,
                include_stop_token=False,
            )
            chosen_target_index = int(target_scores.argmax().item())
            current_selection.append(possible_targets[chosen_target_index])
            if not interaction.is_multi_select():
                possible_targets.remove(current_selection[-1])
                candidates = self._deck_ids_tensor(
                    self._cards_to_deck_ids(possible_targets)
                )

        return current_selection

    def _deck_ids_tensor(self, deck_ids: list[int]) -> torch.Tensor:
        return torch.tensor(
            deck_ids, device=self.selector_device, dtype=torch.long
        )

    def _score_targets(
        self,
        candidates: torch.Tensor,
        current_selection_tensor: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        if self.profiler is not None:
            with self.profiler.timed(self.selector_device) as span:
                scores = self.selector(
                    candidates,
                    current_selection_tensor,
                    transformed_state,
                    embedded_interaction,
                    card_indices,
                    include_stop_token=include_stop_token,
                )
            self.profiler.selector_s += span.elapsed
            self.profiler.selector_calls += 1
            return scores

        return self.selector(
            candidates,
            current_selection_tensor,
            transformed_state,
            embedded_interaction,
            card_indices,
            include_stop_token=include_stop_token,
        )

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
