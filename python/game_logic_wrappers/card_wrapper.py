import csharp_runtime
from gamecore.card import ICard


class CardWrapper:
    card: ICard

    def __init__(self, card: ICard):
        self.card = card

    def to_serializable(self):
        return self.card.ToSerializable()
