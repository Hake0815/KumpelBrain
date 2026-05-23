#ifndef GAME_INTERACTION_EMBEDDING_H
#define GAME_INTERACTION_EMBEDDING_H

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "network/include/AbilityEmbedding.h"
#include "network/include/AttackEmbedding.h"
#include "network/include/CardEmbedding.h"
#include "network/include/ConditionEmbedding.h"
#include "network/include/InstructionDataEmbedding.h"
#include "network/include/InstructionEmbedding.h"
#include "network/include/MultiHeadAttention.h"
#include "network/include/NormalizedLinear.h"
#include "network/include/SaveLoadMixin.h"
#include "network/include/SharedEmbeddingHolder.h"
#include "network/include/SharedInstructionEmbeddings.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufGameInteraction = gamecore::serialization::ProtoBufGameInteraction;

struct FlatConditionalTargetQuery {
    std::vector<int64_t> node_is_leaf;
    std::vector<int64_t> node_logical_operator;
    std::vector<int64_t> node_depth;
    std::vector<int64_t> child_ptr;
    std::vector<int64_t> child_idx;
    std::vector<int64_t> leaf_node_index;
    std::vector<int64_t> leaf_int_range;  // two consecutive ints for min and max
    std::vector<int64_t> leaf_selection_qualifier;
    std::vector<int64_t> root_node_index;
    std::vector<int64_t> root_target_data_index;
};
struct FlatGameInteractionBatch {
    std::vector<int64_t> game_interaction_types;
    std::vector<int64_t> game_interaction_data_types;
    std::vector<int64_t> game_interaction_data_type_offsets;
    std::vector<int64_t> number_data_batch_index;
    std::vector<int64_t> number_data;
    std::vector<int64_t> target_data_batch_index;
    std::vector<int64_t> target_data_possible_targets_deck_ids;
    std::vector<int64_t> target_data_possible_targets_deck_ids_length;
    std::vector<int64_t> target_data_target_action;
    std::vector<int64_t> target_data_remainder_action;
    std::vector<int64_t> target_data_number_of_targets;
    FlatConditionalTargetQuery flat_conditional_target_query;
    std::vector<int64_t> interaction_card_batch_index;
    std::vector<int64_t> interaction_card_deck_id;
    InstructionsAndConditions instructions_and_conditions;
    std::vector<int64_t> select_from_batch_index;
    std::vector<int64_t> select_from;
};

struct FlatConditionalTargetQueryTensors {
    torch::Tensor node_is_leaf;
    torch::Tensor node_logical_operator;
    torch::Tensor node_depth;
    torch::Tensor child_ptr;
    torch::Tensor child_idx;
    torch::Tensor leaf_node_index;
    torch::Tensor leaf_int_range;  // two consecutive ints for min and max
    torch::Tensor leaf_selection_qualifier;
    torch::Tensor root_node_index;
    torch::Tensor root_target_data_index;
};
struct FlatGameInteractionBatchTensors {
    torch::Tensor game_interaction_types;
    torch::Tensor game_interaction_data_types;
    torch::Tensor game_interaction_data_type_offsets;
    torch::Tensor number_data_batch_index;
    torch::Tensor number_data;
    torch::Tensor target_data_batch_index;
    torch::Tensor target_data_possible_targets_deck_ids;
    torch::Tensor target_data_possible_targets_deck_ids_length;
    torch::Tensor target_data_target_action;
    torch::Tensor target_data_remainder_action;
    torch::Tensor target_data_number_of_targets;
    FlatConditionalTargetQueryTensors flat_conditional_target_query;
    torch::Tensor interaction_card_batch_index;
    torch::Tensor interaction_card_deck_id;
    InstructionsAndConditions instructions_and_conditions;
    torch::Tensor select_from_batch_index;
    torch::Tensor select_from;
};

FlatGameInteractionBatchTensors flat_game_interaction_batch_to_tensors(const FlatGameInteractionBatch& flat,
                                                                       torch::Device device = torch::kCPU);

struct GameInteractionEmbeddingImpl : torch::nn::Module, SaveLoadMixin<GameInteractionEmbeddingImpl> {
    GameInteractionEmbeddingImpl(std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
                                 int64_t dimension_out, const SharedInstructionEmbeddings& shared_instruction_embeddings,
                                 torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    /// \param game_interactions The game interactions to embed.
    /// \param card_indices A tensor holding the indices of cards, where the index is the deck id.
    /// \param cards The cards of the state.
    torch::Tensor forward(const std::vector<ProtoBufGameInteraction>& game_interactions, torch::Tensor card_indices,
                          torch::Tensor cards);

   private:
    void register_game_interaction_specific_modules(torch::Device device, torch::Dtype dtype);

    torch::Tensor embed_conditional_target_queries(const FlatConditionalTargetQueryTensors& flat);
    torch::Tensor embed_target_action(const torch::Tensor& target_action);
    torch::Tensor embed_remainder_action(const torch::Tensor& remainder_action);
    torch::Tensor embed_target_data(const FlatGameInteractionBatchTensors& tensors, torch::Tensor card_indices,
                                    torch::Tensor cards);
    torch::Tensor embed_attack_energy_costs(const InstructionsAndConditions& instructions_and_conditions);
    torch::Tensor embed_attacks(const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
                                const std::vector<int64_t>& instruction_attack_indices,
                                const torch::Tensor& attack_energy_costs,
                                const std::vector<int64_t>& energy_slot_per_token);
    torch::Tensor embed_ability(const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
                                const std::vector<int64_t>& instruction_ability_indices,
                                const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
                                const std::vector<int64_t>& ability_condition_row_for_instruction_ability);
    torch::Tensor embed_number_data(const torch::Tensor& number_data);
    torch::Tensor embed_interaction_data(const FlatGameInteractionBatchTensors& flat_tensors,
                                         const torch::Tensor& embedded_target_data,
                                         const torch::Tensor& embedded_number_data,
                                         const torch::Tensor& interaction_card_data,
                                         const torch::Tensor& embedded_attack_data,
                                         const torch::Tensor& embedded_ability_data,
                                         const torch::Tensor& embedded_select_from);
    torch::Tensor reduce_game_interactions(const FlatGameInteractionBatchTensors& flat_tensors,
                                           const torch::Tensor& embedded_interaction_data);

    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::TensorOptions mask_tensor_options_;
    torch::TensorOptions index_tensor_options_;
    torch::TensorOptions float_tensor_options_;
    SharedEmbeddingHolder shared_embedding_holder_{nullptr};
    InstructionDataEmbedding instruction_data_embedding_{nullptr};
    InstructionEmbedding instruction_embedding_{nullptr};
    ConditionEmbedding condition_embedding_{nullptr};
    AttackEmbedding attack_embedding_{nullptr};
    AbilityEmbedding ability_embedding_{nullptr};
    torch::nn::Embedding game_interaction_type_embedding_{nullptr};
    torch::nn::Embedding game_interaction_data_type_embedding_{nullptr};
    NormalizedLinear conditional_query_int_range_embedding_{nullptr};
    torch::nn::Embedding conditional_query_selection_qualifier_embedding_{nullptr};
    torch::nn::Linear conditional_query_leaf_projection_{nullptr};
    torch::nn::Embedding conditional_query_operator_embedding_{nullptr};
    MultiHeadAttention conditional_query_attention_{nullptr};
    MultiHeadAttention target_data_attention_{nullptr};
    MultiHeadAttention game_interaction_attention_{nullptr};
    torch::nn::Embedding action_on_selection_embedding_{nullptr};
    torch::nn::Embedding target_data_addition_embedding_{nullptr};
    torch::nn::Embedding remainder_data_addition_embedding_{nullptr};
    NormalizedLinear number_of_targets_embedding_{nullptr};
    NormalizedLinear number_data_embedding_{nullptr};
    torch::nn::Embedding select_from_embedding_{nullptr};
};

TORCH_MODULE(GameInteractionEmbedding);

#endif
