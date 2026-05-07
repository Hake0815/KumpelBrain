#ifndef GAME_INTERACTION_EMBEDDING_H
#define GAME_INTERACTION_EMBEDDING_H

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "network/include/CardEmbedding.h"
#include "network/include/MultiHeadAttention.h"
#include "network/include/NormalizedLinear.h"
#include "network/include/SaveLoadMixin.h"
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

struct GameInteractionEmbeddingImpl : torch::nn::Module, SaveLoadMixin<GameInteractionEmbeddingImpl> {
    GameInteractionEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                                 torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const std::vector<ProtoBufGameInteraction>& game_interactions);

   private:
    torch::Tensor embed_conditional_target_queries(const FlatConditionalTargetQuery& flat);

    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::nn::Embedding game_interaction_type_embedding_{nullptr};
    NormalizedLinear conditional_query_int_range_embedding_{nullptr};
    torch::nn::Embedding conditional_query_selection_qualifier_embedding_{nullptr};
    torch::nn::Linear conditional_query_leaf_projection_{nullptr};
    torch::nn::Embedding conditional_query_operator_embedding_{nullptr};
    MultiHeadAttention conditional_query_attention_{nullptr};
};

TORCH_MODULE(GameInteractionEmbedding);

#endif
