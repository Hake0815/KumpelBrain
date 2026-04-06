#ifndef CARD_EMBEDDING_H
#define CARD_EMBEDDING_H

#include <ATen/core/TensorBody.h>
#include <torch/nn/modules/embedding.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "../include/ConditionEmbedding.h"
#include "../include/InstructionDataEmbedding.h"
#include "../include/InstructionEmbedding.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include "../src/serialization/gamecore_serialization.pb.h"
#include "network/include/AbilityEmbedding.h"
#include "network/include/AttackEmbedding.h"
#include "network/include/MultiHeadAttention.h"
#include "network/include/NormalizedLinear.h"

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
    std::vector<int64_t> energy_flat;
    std::vector<int64_t> energy_slot_per_token;
};

struct CardFeatures {
    std::vector<int64_t> card_type;
    std::vector<int64_t> card_subtype;
    std::vector<int64_t> energy_type;
    std::vector<uint8_t> energy_type_mask;
    std::vector<int64_t> max_hp;
    std::vector<uint8_t> max_hp_mask;
    std::vector<int64_t> weakness;
    std::vector<uint8_t> weakness_mask;
    std::vector<int64_t> resistance;
    std::vector<uint8_t> resistance_mask;
    std::vector<int64_t> retreat_cost;
    std::vector<uint8_t> retreat_cost_mask;
    std::vector<int64_t> number_of_prize_cards_on_knockout;
    std::vector<uint8_t> number_of_prize_cards_on_knockout_mask;
    std::vector<int64_t> current_damage;
    std::vector<uint8_t> current_damage_mask;

    std::vector<int64_t> flattened_pokemon_turn_traits;
    std::vector<int64_t> pokemon_turn_trait_card_indices;
    std::vector<int64_t> flattened_provided_energies;
    std::vector<int64_t> provided_energy_card_indices;
    std::vector<int64_t> flattened_attached_energies;
    std::vector<int64_t> attached_energy_card_indices;

    InstructionsAndConditions instructions_and_conditions;
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
    NormalizedLinear retreat_cost_embedding_{nullptr};
    NormalizedLinear number_of_prize_cards_on_knockout_embedding_{nullptr};
    NormalizedLinear current_damage_embedding_{nullptr};
    torch::nn::Embedding pokemon_turn_trait_embedding_{nullptr};

    InstructionEmbedding instruction_embedding_{nullptr};
    ConditionEmbedding condition_embedding_{nullptr};
    SharedEmbeddingHolder shared_embedding_holder_{nullptr};
    InstructionDataEmbedding instruction_data_embedding_{nullptr};
    MultiHeadAttention card_instructions_multi_head_attention_{nullptr};
    torch::nn::Embedding card_instruction_query_embedding_{nullptr};
    MultiHeadAttention card_conditions_multi_head_attention_{nullptr};
    torch::nn::Embedding card_condition_query_embedding_{nullptr};

    MultiHeadAttention card_self_multi_head_attention_{nullptr};
    MultiHeadAttention card_pooling_multi_head_attention_{nullptr};
    torch::nn::Embedding card_pooling_query_embedding_{nullptr};

    CardFeatures collect_card_features(const std::vector<ProtoBufCard>& card_batch);

    void collect_instructions_and_conditions(const std::vector<ProtoBufCard>& card_batch,
                                             InstructionsAndConditions& instructions_and_conditions, int card_index);

    std::pair<torch::Tensor, torch::Tensor> embed_card_features(const CardFeatures& card_features, int batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_flattened_card_feature(
        torch::nn::Embedding& embedding, const std::vector<int64_t>& flattened_card_feature,
        const std::vector<int64_t>& card_indices, int batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_instructions_and_conditions(
        const InstructionsAndConditions& instructions_and_conditions, int batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_attacks(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
        const std::vector<int64_t>& instruction_attack_indices, const std::vector<int64_t>& energy_flat,
        const std::vector<int64_t>& energy_slot_per_token,
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
