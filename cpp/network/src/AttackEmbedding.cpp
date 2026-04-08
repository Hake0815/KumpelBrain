#include "../include/AttackEmbedding.h"

#include <ATen/ops/zeros.h>

#include "network/include/AttentionUtils.h"

AttackEmbeddingImpl::AttackEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    attack_query_embedding_ = register_module("attack_query_embedding", torch::nn::Embedding(1, dimension_out_));
    multi_head_attention_ =
        register_module("multi_head_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0, true, device_, dtype_));
    to(device_, dtype_);
}

torch::Tensor AttackEmbeddingImpl::forward(const torch::Tensor& attack_energy_costs,
                                           const torch::Tensor& embedded_instructions,
                                           const torch::Tensor& instruction_valid_token_mask) {
    auto attack_query = attack_query_embedding_
                            ->forward(torch::zeros({attack_energy_costs.size(0)},
                                                   torch::TensorOptions().device(device_).dtype(torch::kLong)))
                            .unsqueeze(1);

    auto tokens = torch::cat({attack_energy_costs.unsqueeze(1), embedded_instructions}, 1);
    auto token_mask = torch::cat(
        {torch::ones({attack_energy_costs.size(0), 1}, torch::TensorOptions().device(device_).dtype(torch::kBool)),
         instruction_valid_token_mask},
        1);
    return attention_utils::masked_attention_pooling(multi_head_attention_, attack_query, tokens, token_mask);
}