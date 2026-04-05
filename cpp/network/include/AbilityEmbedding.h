#ifndef ABILITY_EMBEDDING_H
#define ABILITY_EMBEDDING_H

#include "../include/MultiHeadAttention.h"
#include "../include/SaveLoadMixin.h"
#include "../src/serialization/gamecore_serialization.pb.h"

using ProtoBufAttack = gamecore::serialization::ProtoBufAttack;

struct AbilityEmbeddingImpl : torch::nn::Module, SaveLoadMixin<AbilityEmbeddingImpl> {
    AbilityEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    /**
    embedded_instructions: (B, L_I, D)
    instruction_valid_token_mask: (B, L_I), bool, same B and device as the tensors above
    embedded_conditions: (B, L_C, D)
    condition_valid_token_mask: (B, L_C), bool, same B and device as the tensors above
    B...: batch size
    D...: dimension
    L...: sequence length
    */
    torch::Tensor forward(const torch::Tensor& embedded_instructions, const torch::Tensor& instruction_valid_token_mask,
                          const torch::Tensor& embedded_conditions, const torch::Tensor& condition_valid_token_mask);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::nn::Embedding ability_instruction_query_embedding_{nullptr};
    torch::nn::Embedding ability_condition_query_embedding_{nullptr};
    MultiHeadAttention instruction_multi_head_attention_{nullptr};
    MultiHeadAttention condition_multi_head_attention_{nullptr};
};

TORCH_MODULE(AbilityEmbedding);

#endif
