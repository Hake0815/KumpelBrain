import torch
import torch.nn as nn
from action_scores import ActionScores
from save_load_mixin import SaveLoadMixin
from multi_head_attention import MultiHeadAttentionArgs
from scoring_block import CrossAttentionScoringBlock


class InteractionNetwork(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_inner: int,
        attention_args: MultiHeadAttentionArgs,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.scoring_block = CrossAttentionScoringBlock(
            dimension_out,
            dimension_inner,
            attention_args,
            include_pre_ffn=True,
            **factory_kwargs,
        )

    def forward(
        self,
        embedded_interactions: torch.Tensor,
        state: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> ActionScores:
        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask.unsqueeze(1).expand(
                embedded_interactions.size(0),
                embedded_interactions.size(1),
                state.size(1),
            )
        return self.scoring_block(
            embedded_interactions, state, state, attn_mask=attn_mask
        )
