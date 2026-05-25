import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from state_transformer import StateTransformer
from kumpel_embedding import GameEmbedding
from game_embedding import extract_card_embeddings
from interaction_network import InteractionNetwork
from selector import Selector
from multi_head_attention import MultiHeadAttentionArgs


class KumpelNetwork(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_state_inner: int,
        dimension_interaction_inner: int,
        state_attention_args: MultiHeadAttentionArgs,
        interaction_attention_args: MultiHeadAttentionArgs,
        target_attention_args: MultiHeadAttentionArgs,
        dimension_target_inner: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.state_transformer = StateTransformer(
            dimension_out,
            dimension_state_inner,
            state_attention_args,
            num_layers,
            **self.factory_kwargs
        )
        self.game_embedding = GameEmbedding(dimension_out, torch.device("cpu"), dtype)
        self.interaction_network = InteractionNetwork(
            dimension_out,
            dimension_interaction_inner,
            interaction_attention_args,
            **self.factory_kwargs
        )
        self.selector = Selector(
            dimension_out,
            dimension_target_inner,
            target_attention_args,
            **self.factory_kwargs,
        )

    def forward(
        self, game_state: bytes, game_interactions: list[bytes]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compute_device = self.factory_kwargs["device"]
        embedded_game_state, card_indices = self.game_embedding.embedGameState(
            game_state
        )
        embedded_game_state = embedded_game_state.to(compute_device)

        transformed_state = self.state_transformer(embedded_game_state.unsqueeze(0))
        transformed_cards_cpu = extract_card_embeddings(
            transformed_state.squeeze(0)
        ).cpu()

        # embedGameInteraction requires deck_ids, card_indices, and cards on GameEmbedding device (CPU).
        embedded_interactions = self.game_embedding.embedGameInteraction(
            game_interactions, card_indices, transformed_cards_cpu
        ).to(compute_device)
        card_indices = card_indices.to(compute_device)

        interaction_scores = self.interaction_network(
            embedded_interactions.unsqueeze(0), transformed_state
        )
        return (
            interaction_scores.squeeze(0),
            transformed_state.squeeze(0),
            embedded_interactions,
            card_indices,
        )

    def select_target(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        return self.selector(
            candidates,
            partial_selection,
            transformed_state,
            embedded_interaction,
            card_indices,
            include_stop_token,
        ).squeeze(0)
