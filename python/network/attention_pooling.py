import torch
from multi_head_attention import MultiHeadAttention


def query_sum_attention_pooling(
    multi_head_attention: MultiHeadAttention,
    query: torch.Tensor,
    tokens: list[torch.Tensor],
) -> torch.Tensor:
    if len(tokens) == 0:
        return torch.empty(
            (0, query.shape[-1]),
            device=query.device,
            dtype=query.dtype,
        )
    return attention_pooling(multi_head_attention, query, tokens)+query.squeeze(1)

def attention_pooling(
    multi_head_attention: MultiHeadAttention,
    query: torch.Tensor,
    tokens: list[torch.Tensor],
) -> torch.Tensor:
    if len(tokens) == 0:
        return torch.empty(
            (0, query.shape[-1]),
            device=query.device,
            dtype=query.dtype,
        )

    padded_tokens, valid_token_mask = pad_sequence_list(tokens, query.shape[-1])
    return masked_attention_pooling(multi_head_attention, query, padded_tokens, valid_token_mask)


def pad_sequence_list(
    sequence_list: list[torch.Tensor], dimension_out: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequence_list:
        raise ValueError("pad_sequence_list requires at least one sequence")

    batch_size = len(sequence_list)
    max_sequence_length = max(sequence.shape[0] for sequence in sequence_list)
    device = sequence_list[0].device
    dtype = sequence_list[0].dtype

    padded_sequences = torch.zeros(
        (batch_size, max_sequence_length, dimension_out), device=device, dtype=dtype
    )
    valid_token_mask = torch.zeros(
        (batch_size, max_sequence_length), device=device, dtype=torch.bool
    )

    for batch_index, sequence in enumerate(sequence_list):
        sequence_length = sequence.shape[0]
        if sequence_length == 0:
            continue
        padded_sequences[batch_index, :sequence_length] = sequence
        valid_token_mask[batch_index, :sequence_length] = True

    return padded_sequences, valid_token_mask


def masked_attention_pooling(
    multi_head_attention: MultiHeadAttention,
    query: torch.Tensor,
    padded_sequences: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    if padded_sequences.shape[1] == 0:
        return torch.zeros(
            (padded_sequences.shape[0], padded_sequences.shape[-1]),
            device=padded_sequences.device,
            dtype=padded_sequences.dtype,
        )

    attention_mask = make_padding_attention_mask(valid_token_mask, padded_sequences.dtype, query.shape[1]) 
    has_any_valid_token = valid_token_mask.any(dim=1).unsqueeze(1)

    return multi_head_attention(
        query,
        padded_sequences,
        padded_sequences,
        attn_mask=attention_mask,
    ).squeeze(1) * has_any_valid_token

def make_padding_attention_mask(
    valid_token_mask: torch.Tensor,
    dtype: torch.dtype,
    query_seq_len: int,
) -> torch.Tensor:
    batch_size, seq_len_kv = valid_token_mask.shape
    attention_mask = torch.zeros(
        (batch_size, query_seq_len, seq_len_kv),
        device=valid_token_mask.device,
        dtype=dtype,
    )
    invalid_key_mask = (
        (~valid_token_mask).unsqueeze(1).expand(-1, query_seq_len, -1)
    )
    return attention_mask.masked_fill(
        invalid_key_mask, torch.finfo(dtype).min
    )
