import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from state_transformer import StateTransformer
from kumpel_embedding import GameEmbedding
from interaction_network import InteractionNetwork


class KumpelNetwork(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_state_inner: int,
        dimension_interaction_inner: int,
        dimension_head: int,
        nheads: int,
        num_layers: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.state_transformer = StateTransformer(
            dimension_out,
            dimension_state_inner,
            dimension_head,
            nheads,
            num_layers,
            dropout,
            bias,
            **factory_kwargs
        )
        self.game_embedding = GameEmbedding(dimension_out, torch.device("cpu"), dtype)
        self.interaction_network = InteractionNetwork(
            dimension_out,
            dimension_interaction_inner,
            dimension_head,
            nheads,
            dropout,
            bias,
            **factory_kwargs
        )

    def forward(
        self, game_state: bytes, game_interactions: list[bytes]
    ) -> torch.Tensor:
        embedded_game_state, card_indices = self.game_embedding.embedGameState(
            game_state
        )
   
        transformed_state = self.state_transformer(embedded_game_state.unsqueeze(0))
        transformed_cards = transformed_state.squeeze(0)[2:]
    
        embedded_interactions = self.game_embedding.embedGameInteraction(
            game_interactions, card_indices, transformed_cards
        )
        interaction_scores = self.interaction_network(
            embedded_interactions.unsqueeze(0), transformed_state
        )
        return interaction_scores.squeeze(0)
