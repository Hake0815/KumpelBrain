#ifndef CARD_EMBEDDING_H
#define CARD_EMBEDDING_H

#include <ATen/core/TensorBody.h>
#include <torch/nn/modules/embedding.h>

#include <utility>

#include "../include/ConditionEmbedding.h"
#include "../include/InstructionDataEmbedding.h"
#include "../include/InstructionEmbedding.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include "../src/serialization/gamecore_serialization.pb.h"
#include "network/include/AbilityEmbedding.h"
#include "network/include/AttackEmbedding.h"
#include "network/include/MultiHeadAttention.h"

using ProtoBufCard = gamecore::serialization::ProtoBufCard;

struct InstructionsAndConditions {
    std::vector<std::vector<ProtoBufInstruction>> instructions;
    std::vector<std::vector<ProtoBufCondition>> conditions;
    std::vector<std::pair<int, int>> instruction_card_parent_indices;
    std::vector<std::pair<int, int>> condition_card_parent_indices;
    std::vector<int64_t> instruction_card_indices;
    std::vector<int64_t> instruction_ability_indices;
    std::vector<int64_t> instruction_attack_indices;
    std::vector<int64_t> condition_card_indices;
    /// Same length as instruction_ability_indices: global condition row index for that ability's
    /// instructions, or -1 if the ability has no conditions.
    std::vector<int64_t> ability_condition_row_for_instruction_ability;
    std::vector<torch::Tensor> attack_energy_costs;
};

struct CardEmbeddingImpl : torch::nn::Module, SaveLoadMixin<CardEmbeddingImpl> {
    CardEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const std::vector<ProtoBufCard>& card_batch);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    AbilityEmbedding ability_embedding_{nullptr};
    AttackEmbedding attack_embedding_{nullptr};

    InstructionEmbedding instruction_embedding_{nullptr};
    ConditionEmbedding condition_embedding_{nullptr};
    SharedEmbeddingHolder shared_embedding_holder_{nullptr};
    InstructionDataEmbedding instruction_data_embedding_{nullptr};
    MultiHeadAttention card_instructions_multi_head_attention_{nullptr};
    torch::nn::Embedding card_instruction_query_embedding_{nullptr};
    MultiHeadAttention card_conditions_multi_head_attention_{nullptr};
    torch::nn::Embedding card_condition_query_embedding_{nullptr};

    MultiHeadAttention card_token_multi_head_attention_{nullptr};
    torch::nn::Embedding card_token_query_embedding_{nullptr};

    InstructionsAndConditions collect_instructions_and_conditions(const std::vector<ProtoBufCard>& card_batch);
    std::pair<torch::Tensor, torch::Tensor> embed_instructions_and_conditions(
        const InstructionsAndConditions& instructions_and_conditions, int batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_attacks(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
        const std::vector<int64_t>& instruction_attack_indices, const std::vector<torch::Tensor>& attack_energy_costs,
        const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_ability(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
        const std::vector<int64_t>& instruction_ability_indices,
        const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
        const std::vector<int64_t>& ability_condition_row_for_instruction_ability,
        const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size);

    std::pair<torch::Tensor, torch::Tensor> pad_to_batch(const std::vector<int64_t>& card_indices,
                                                         const std::vector<std::pair<int, int>>& card_parent_indices,
                                                         int batch_size, const torch::Tensor& pooled_instructions);

    std::pair<torch::Tensor, torch::Tensor> embed_card_instructions(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
        const std::vector<int64_t>& instruction_card_indices,
        const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_card_conditions(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
        const std::vector<int64_t>& condition_card_indices,
        const std::vector<std::pair<int, int>>& condition_card_parent_indices, int batch_size);
};

TORCH_MODULE(CardEmbedding);

#endif
