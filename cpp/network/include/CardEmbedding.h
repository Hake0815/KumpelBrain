#ifndef CARD_EMBEDDING_H
#define CARD_EMBEDDING_H

#include <ATen/core/TensorBody.h>
#include <torch/nn/modules/embedding.h>

#include <cstdint>
#include <tuple>
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

using ProtoBufCardState = gamecore::serialization::ProtoBufCardState;
using ProtoBufCard = gamecore::serialization::ProtoBufCard;

struct ParentIndex {
    int card;
    int slot;
};

struct InstructionsAndConditions {
    std::vector<std::vector<ProtoBufInstruction>> instructions;
    std::vector<std::vector<ProtoBufCondition>> conditions;
    std::vector<ParentIndex> instruction_card_parent_indices;
    std::vector<ParentIndex> condition_card_parent_indices;
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

struct AdjacencyMatrices {
    /// Sparse COO float tensor of shape [num_cards, num_cards]; nonzero at (child, parent) when child evolves from
    /// parent name in the same player's deck.
    torch::Tensor evolves_from_adjacency;
    /// Sparse COO float tensor of shape [num_cards, num_cards]; nonzero at (host, energy_batch_index) when the host
    /// lists that energy card's deck_id in attached_energy_cards (resolved after one pass over the batch).
    torch::Tensor attached_energy_adjacency;
    /// Sparse COO float tensor of shape [num_cards, num_cards]; nonzero at (host, pre_evolution_batch_index) when the
    /// host lists that pre-evolution card's deck_id in pre_evolution_ids (resolved after one pass over the batch).
    torch::Tensor pre_evolutions_adjacency;
};
struct CardFeatures {
    std::vector<int64_t> card_type;
    std::vector<int64_t> card_subtype;
    std::vector<int64_t> energy_type;
    std::vector<int64_t> energy_type_context;
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

    AdjacencyMatrices adjacency_matrices;
    InstructionsAndConditions instructions_and_conditions;
};

/// Staged H2D buffers. All int64 scalar/index vectors packed into `int64_buf`;
/// all uint8 mask vectors packed into `uint8_buf` (cast to bool on upload).
/// Per-field views are exposed as tensors already sliced from the device buffer.
struct StagedTensors {
    torch::Tensor card_type;
    torch::Tensor card_subtype;
    torch::Tensor max_hp;
    torch::Tensor retreat_cost;
    torch::Tensor number_of_prize_cards_on_knockout;
    torch::Tensor current_damage;
    torch::Tensor flattened_pokemon_turn_traits;
    torch::Tensor pokemon_turn_trait_card_indices;
    torch::Tensor provided_energy_card_indices;
    torch::Tensor attached_energy_card_indices;
    torch::Tensor energy_type_indices;
    torch::Tensor energy_type_contexts;

    torch::Tensor energy_type_mask;
    torch::Tensor weakness_mask;
    torch::Tensor resistance_mask;
    torch::Tensor max_hp_mask;
    torch::Tensor retreat_cost_mask;
    torch::Tensor number_of_prize_cards_on_knockout_mask;
    torch::Tensor current_damage_mask;
};

/// Embeds a batch of `ProtoBufCard` into shape [batch, dimension_out].
struct CardEmbeddingImpl : torch::nn::Module, SaveLoadMixin<CardEmbeddingImpl> {
    CardEmbeddingImpl(std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out,
                      torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    std::pair<torch::Tensor, AdjacencyMatrices> forward(const std::vector<ProtoBufCardState>& card_batch);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::TensorOptions mask_tensor_options_;
    torch::TensorOptions index_tensor_options_;
    torch::TensorOptions float_tensor_options_;
    torch::Tensor ones_1x1_bool_;
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

    CardFeatures collect_card_features(const std::vector<ProtoBufCardState>& card_batch);

    void append_card_instructions_and_conditions(const ProtoBufCard& card,
                                                 InstructionsAndConditions& instructions_and_conditions,
                                                 int64_t card_index);

    StagedTensors stage_features(const CardFeatures& card_features);

    std::pair<torch::Tensor, torch::Tensor> embed_card_features(const CardFeatures& card_features,
                                                                const StagedTensors& staged, int64_t batch_size);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> embed_energy_type_features(
        const CardFeatures& card_features, const StagedTensors& staged, int64_t batch_size);
    std::pair<torch::Tensor, torch::Tensor> combine_flat_embedded_card_feature(
        const torch::Tensor& flat_embedded_feature, const std::vector<int64_t>& card_indices,
        const torch::Tensor& card_indices_tensor, int64_t batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_instructions_and_conditions(
        const InstructionsAndConditions& instructions_and_conditions, const torch::Tensor& attack_energy_costs,
        int64_t batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_attacks(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
        const std::vector<int64_t>& instruction_attack_indices, const torch::Tensor& attack_energy_costs,
        const std::vector<int64_t>& energy_slot_per_token,
        const std::vector<ParentIndex>& instruction_card_parent_indices, int64_t batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_ability(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
        const std::vector<int64_t>& instruction_ability_indices,
        const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
        const std::vector<int64_t>& ability_condition_row_for_instruction_ability,
        const std::vector<ParentIndex>& instruction_card_parent_indices, int64_t batch_size);

    std::pair<torch::Tensor, torch::Tensor> pad_to_batch(const std::vector<int64_t>& card_indices,
                                                         const std::vector<ParentIndex>& card_parent_indices,
                                                         int64_t batch_size, const torch::Tensor& pooled_tokens);

    std::pair<torch::Tensor, torch::Tensor> embed_card_instructions(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
        const std::vector<int64_t>& instruction_card_indices,
        const std::vector<ParentIndex>& instruction_card_parent_indices, int64_t batch_size);
    std::pair<torch::Tensor, torch::Tensor> embed_card_conditions(
        const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
        const std::vector<int64_t>& condition_card_indices,
        const std::vector<ParentIndex>& condition_card_parent_indices, int64_t batch_size);
};

TORCH_MODULE(CardEmbedding);

#endif
