import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from transformer_layer import TransformerLayer
from multi_head_attention import MultiHeadAttentionArgs

class StateTransformer(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_inner: int,
        attention_args: MultiHeadAttentionArgs,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(dimension_out, dimension_inner, attention_args, **factory_kwargs) for _ in range(num_layers)])

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).expand(
                x.size(0), x.size(1), x.size(1)
            )
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x
