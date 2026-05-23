import torch
import torch.nn as nn
from save_load_mixin import SaveLoadMixin


class FeedForward(nn.Module, SaveLoadMixin):
    def __init__(
        self,
        dimension_out: int,
        dimension_inner: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.linear_in = nn.Linear(dimension_out, dimension_inner, **factory_kwargs)
        self.linear_out = nn.Linear(dimension_inner, dimension_out, **factory_kwargs)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.activation(self.linear_in(x)))