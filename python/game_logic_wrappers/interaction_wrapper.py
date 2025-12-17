import csharp_runtime
import clr

from card_wrapper import CardWrapper
from gamecore.game.interaction import (
    GameInteraction,
    ConditionalTargetData,
    TargetData,
    WinnerData,
    GameInteractionDataType
)
from gamecore.card import ICard
from System.Collections.Generic import List


class InteractionWrapper:
    interaction: GameInteraction

    def __init__(self, interaction: GameInteraction):
        self.interaction = interaction

    def get_type(self) -> str:
        return self.interaction.Type

    def is_with_target(self) -> bool:
        return self.interaction.GameControllerMethodWithTargets is not None

    def is_with_condition_target(self) -> bool:
        return self.is_with_target() and self.interaction.Data.ContainsKey(
            GameInteractionDataType.ConditionalTargetData
        )

    def get_targets(self) -> list[CardWrapper]:
        if not self.is_with_target():
            return []

        if self.is_with_condition_target():
            return [
                CardWrapper(card)
                for card in ConditionalTargetData(
                    self.interaction.Data[GameInteractionDataType.ConditionalTargetData]
                ).PossibleTargets
            ]

        return [
            CardWrapper(card)
            for card in TargetData(
                self.interaction.Data[GameInteractionDataType.TargetData]
            ).PossibleTargets
        ]

    def is_target_condition_fulfilled(self, targets: list[CardWrapper]) -> bool:
        condition = ConditionalTargetData(
            self.interaction.Data[GameInteractionDataType.ConditionalTargetData]
        ).ConditionalTargetQuery
        card_list = List[ICard]()
        for card_wrapper in targets:
            card_list.Add(card_wrapper.card)
        return condition.IsMet(card_list)

    def get_number_of_targets(self) -> int:
        return TargetData(self.interaction.Data[GameInteractionDataType.TargetData]).NumberOfTargets

    def perform_action(self) -> None:
        self.interaction.GameControllerMethod.Invoke()

    def perform_action_with_targets(self, targets: list[CardWrapper]) -> None:
        card_list = List[ICard]()
        for card_wrapper in targets:
            card_list.Add(card_wrapper.card)

        self.interaction.GameControllerMethodWithTargets.Invoke(card_list)

    def is_game_over(self) -> bool:
        return self.interaction.Data.ContainsKey(GameInteractionDataType.WinnerData)

    def get_game_over_message(self) -> str:
        return WinnerData(self.interaction.Data[GameInteractionDataType.WinnerData]).Message

    def try_cast(self, T, obj):
        return T(obj) if clr.GetClrType(T).IsInstanceOfType(obj) else None
