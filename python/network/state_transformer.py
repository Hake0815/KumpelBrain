import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin
from transformer_layer import TransformerLayer

class StateTransformer(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_inner: int,
        dimension_head: int,
        nheads: int,
        num_layers: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(dimension_out, dimension_inner, dimension_head, nheads, dropout, bias, **factory_kwargs) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x