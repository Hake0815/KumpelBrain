import torch
import torch.nn as nn
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
        self, embedded_interactions: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return self.scoring_block(embedded_interactions, state, state)
