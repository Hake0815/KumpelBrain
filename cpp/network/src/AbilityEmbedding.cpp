#include "../include/AbilityEmbedding.h"

#include <ATen/ops/zeros.h>

#include "network/include/AttentionUtils.h"

AbilityEmbeddingImpl::AbilityEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    ability_instruction_query_embedding_ =
        register_module("ability_instruction_query_embedding", torch::nn::Embedding(1, dimension_out_));
    ability_condition_query_embedding_ =
        register_module("ability_condition_query_embedding", torch::nn::Embedding(1, dimension_out_));
    instruction_multi_head_attention_ =
        register_module("instruction_multi_head_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0, true, device_, dtype_));
    condition_multi_head_attention_ =
        register_module("condition_multi_head_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0, true, device_, dtype_));
    to(device_, dtype_);
}

torch::Tensor AbilityEmbeddingImpl::forward(const torch::Tensor& embedded_instructions,
                                            const torch::Tensor& instruction_valid_token_mask,
                                            const torch::Tensor& embedded_conditions,
                                            const torch::Tensor& condition_valid_token_mask) {
    auto instruction_query = ability_instruction_query_embedding_
                                 ->forward(torch::zeros({embedded_instructions.size(0)},
                                                        torch::TensorOptions().device(device_).dtype(torch::kLong)))
                                 .unsqueeze(1);
    auto condition_query = ability_condition_query_embedding_
                               ->forward(torch::zeros({embedded_conditions.size(0)},
                                                      torch::TensorOptions().device(device_).dtype(torch::kLong)))
                               .unsqueeze(1);
    return attention_utils::masked_attention_pooling(instruction_multi_head_attention_, instruction_query,
                                                     embedded_instructions, instruction_valid_token_mask) +
           attention_utils::masked_attention_pooling(condition_multi_head_attention_, condition_query,
                                                     embedded_conditions, condition_valid_token_mask);
}