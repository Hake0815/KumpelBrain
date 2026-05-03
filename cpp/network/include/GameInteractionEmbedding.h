#ifndef GAME_INTERACTION_EMBEDDING_H
#define GAME_INTERACTION_EMBEDDING_H

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "network/include/CardEmbedding.h"
#include "network/include/SaveLoadMixin.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufGameInteraction = gamecore::serialization::ProtoBufGameInteraction;

struct FlatConditionalTargetQuery {
    std::vector<int64_t> logical_operator;
    std::vector<int64_t> int_range;  // two consecutive ints for min and max
    std::vector<int64_t> selection_qualifier;
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
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::nn::Embedding game_interaction_type_embedding_{nullptr};
};

TORCH_MODULE(GameInteractionEmbedding);

#endif
