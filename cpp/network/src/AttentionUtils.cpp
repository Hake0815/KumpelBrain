#include "../include/AttentionUtils.h"

#include "../include/TensorUtils.h"

namespace attention_utils {

torch::Tensor masked_self_attention_reduce(MultiHeadAttention& multi_head_attention,
                                           const torch::Tensor& padded_sequences,
                                           const torch::Tensor& valid_token_mask) {
    auto attended = masked_self_attention(multi_head_attention, padded_sequences, valid_token_mask);
    return tensor_utils::masked_sequence_sum(attended, valid_token_mask);
}

torch::Tensor masked_self_attention(MultiHeadAttention& multi_head_attention, const torch::Tensor& padded_sequences,
                                    const torch::Tensor& valid_token_mask) {
    if (padded_sequences.size(1) == 0) {
        return torch::zeros({padded_sequences.size(0), padded_sequences.size(2)},
                            torch::TensorOptions().device(padded_sequences.device()).dtype(padded_sequences.dtype()));
    }

    auto attention_mask = tensor_utils::make_padding_attention_mask(valid_token_mask, valid_token_mask.size(1),
                                                                    padded_sequences.scalar_type());
    return padded_sequences +
           multi_head_attention(padded_sequences, padded_sequences, padded_sequences, attention_mask);
}

torch::Tensor masked_attention_pooling(MultiHeadAttention& multi_head_attention, const torch::Tensor& query,
                                       const torch::Tensor& padded_sequences, const torch::Tensor& valid_token_mask) {
    if (padded_sequences.size(1) == 0) {
        return torch::zeros({padded_sequences.size(0), padded_sequences.size(2)},
                            torch::TensorOptions().device(padded_sequences.device()).dtype(padded_sequences.dtype()));
    }
    auto attention_mask =
        tensor_utils::make_padding_attention_mask(valid_token_mask, query.size(1), padded_sequences.scalar_type());
    auto out = multi_head_attention(query, padded_sequences, padded_sequences, attention_mask).squeeze(1);
    auto row_has_key = valid_token_mask.any(1).to(out.dtype()).unsqueeze(1);
    return out * row_has_key;
}

torch::Tensor query_sum_attention_pooling(MultiHeadAttention& multi_head_attention, const torch::Tensor& query,
                                          const torch::Tensor& padded_sequences,
                                          const torch::Tensor& valid_token_mask) {
    if (padded_sequences.size(1) == 0) {
        return torch::zeros({padded_sequences.size(0), padded_sequences.size(2)},
                            torch::TensorOptions().device(padded_sequences.device()).dtype(padded_sequences.dtype()));
    }

    return masked_attention_pooling(multi_head_attention, query, padded_sequences, valid_token_mask) + query.squeeze(1);
}

}  // namespace attention_utils
