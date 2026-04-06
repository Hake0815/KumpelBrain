#ifndef ATTENTION_UTILS_H
#define ATTENTION_UTILS_H

#include <torch/torch.h>

#include "../include/MultiHeadAttention.h"

namespace attention_utils {

torch::Tensor masked_self_attention_reduce(MultiHeadAttention& multi_head_attention,
                                           const torch::Tensor& padded_sequences,
                                           const torch::Tensor& valid_token_mask);

torch::Tensor masked_attention_pooling(MultiHeadAttention& multi_head_attention, const torch::Tensor& query,
                                       const torch::Tensor& padded_sequences, const torch::Tensor& valid_token_mask);
torch::Tensor query_sum_attention_pooling(MultiHeadAttention& multi_head_attention, const torch::Tensor& query,
                                          const torch::Tensor& padded_sequences, const torch::Tensor& valid_token_mask);
}  // namespace attention_utils

#endif
