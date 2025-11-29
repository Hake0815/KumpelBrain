import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, **factory_kwargs).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(1, d_model + 1, **factory_kwargs)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model, **factory_kwargs)
        pe[0, :, 0::2] = torch.sin(position * div_term[0::2], **factory_kwargs)
        pe[0, :, 1::2] = torch.cos(position * div_term[1::2], **factory_kwargs)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
