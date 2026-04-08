#ifndef ATTACK_EMBEDDING_H
#define ATTACK_EMBEDDING_H

#include "../include/MultiHeadAttention.h"
#include "../include/SaveLoadMixin.h"
#include "../src/serialization/gamecore_serialization.pb.h"

using ProtoBufAttack = gamecore::serialization::ProtoBufAttack;

struct AttackEmbeddingImpl : torch::nn::Module, SaveLoadMixin<AttackEmbeddingImpl> {
    AttackEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    /**
    attack_energy_costs: (B, D), sum of energy types
    embedded_instructions: (B, L, D)
    instruction_valid_token_mask: (B, L), bool, same B and device as the tensors above
    B...: batch size
    D...: dimension
    L...: sequence length
    */
    torch::Tensor forward(const torch::Tensor& attack_energy_costs, const torch::Tensor& embedded_instructions,
                          const torch::Tensor& instruction_valid_token_mask);

   private:
    torch::nn::Embedding attack_query_embedding_{nullptr};
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    MultiHeadAttention multi_head_attention_{nullptr};
};

TORCH_MODULE(AttackEmbedding);

#endif
