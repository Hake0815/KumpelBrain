#include "../include/AttentionUtils.h"
#include "../include/TensorUtils.h"

namespace attention_utils {

torch::Tensor masked_self_attention_reduce(
    MultiHeadAttention &multi_head_attention, const torch::Tensor &padded_sequences,
    const torch::Tensor &valid_token_mask) {
  if (padded_sequences.size(1) == 0) {
    return torch::zeros(
        {padded_sequences.size(0), padded_sequences.size(2)},
        torch::TensorOptions()
            .device(padded_sequences.device())
            .dtype(padded_sequences.dtype()));
  }

  auto attention_mask = tensor_utils::make_padding_attention_mask(
      valid_token_mask, padded_sequences.scalar_type());
  auto attended = padded_sequences + multi_head_attention(
                                       padded_sequences, padded_sequences,
                                       padded_sequences, attention_mask);
  return tensor_utils::masked_sequence_sum(attended, valid_token_mask);
}

} // namespace attention_utils
