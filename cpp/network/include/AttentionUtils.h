#ifndef ATTENTION_UTILS_H
#define ATTENTION_UTILS_H

#include "../include/MultiHeadAttention.h"
#include <torch/torch.h>

namespace attention_utils {

torch::Tensor masked_self_attention_reduce(
    MultiHeadAttention &multi_head_attention, const torch::Tensor &padded_sequences,
    const torch::Tensor &valid_token_mask);

} // namespace attention_utils

#endif
