import csharp_runtime
from gamecore.card import ICard
from System.Collections.Generic import List


class CardWrapper:
    card: ICard

    def __init__(self, card: ICard):
        self.card = card

    def to_serializable(self):
        return self.card.ToSerializable()

    def get_deck_id(self) -> int:
        return self.card.DeckId


def convert_card_wrapper_list(card_wrappers: list[CardWrapper]) -> List[ICard]:
    card_list = List[ICard]()
    for card_wrapper in card_wrappers:
        card_list.Add(card_wrapper.card)
    return card_list
