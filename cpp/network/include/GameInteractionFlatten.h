#ifndef GAME_INTERACTION_FLATTEN_H
#define GAME_INTERACTION_FLATTEN_H

#include <torch/torch.h>
#include <vector>

#include "network/include/InstructionsAndConditions.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufGameInteraction = gamecore::serialization::ProtoBufGameInteraction;

struct FlatConditionalTargetQuery {
    std::vector<int64_t> node_is_leaf;
    std::vector<int64_t> node_logical_operator;
    std::vector<int64_t> node_depth;
    std::vector<int64_t> child_ptr;
    std::vector<int64_t> child_idx;
    std::vector<int64_t> leaf_node_index;
    std::vector<int64_t> leaf_int_range;
    std::vector<int64_t> leaf_selection_qualifier;
    std::vector<int64_t> root_node_index;
    std::vector<int64_t> root_target_data_index;
};

struct FlatGameInteractionBatch {
    /// Game index in the outer batch for each flattened interaction.
    std::vector<int64_t> interaction_game_indices;
    /// Cumulative interaction counts per game; size B+1.
    std::vector<int64_t> interaction_segment_offsets;
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
    std::vector<int64_t> target_data_allow_multiple_times;
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
    torch::Tensor leaf_int_range;
    torch::Tensor leaf_selection_qualifier;
    torch::Tensor root_node_index;
    torch::Tensor root_target_data_index;
};

struct FlatGameInteractionBatchTensors {
    torch::Tensor interaction_game_indices;
    torch::Tensor interaction_segment_offsets;
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
    torch::Tensor target_data_allow_multiple_times;
    FlatConditionalTargetQueryTensors flat_conditional_target_query;
    torch::Tensor interaction_card_batch_index;
    torch::Tensor interaction_card_deck_id;
    InstructionsAndConditions instructions_and_conditions;
    torch::Tensor select_from_batch_index;
    torch::Tensor select_from;
};

FlatGameInteractionBatch flatten_game_interaction_batch(
    const std::vector<std::vector<ProtoBufGameInteraction>>& game_interactions_per_game);

FlatGameInteractionBatchTensors flat_game_interaction_batch_to_tensors(const FlatGameInteractionBatch& flat,
                                                                       torch::Device device = torch::kCPU);

#endif
