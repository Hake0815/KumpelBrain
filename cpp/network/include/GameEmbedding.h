#ifndef GAME_EMBEDDING_H
#define GAME_EMBEDDING_H

#include <torch/torch.h>

#include <vector>

#include "network/include/AbilityEmbedding.h"
#include "network/include/AttackEmbedding.h"
#include "network/include/ConditionEmbedding.h"
#include "network/include/GameInteractionEmbedding.h"
#include "network/include/GameStateEmbedding.h"
#include "network/include/InstructionDataEmbedding.h"
#include "network/include/InstructionEmbedding.h"
#include "network/include/SaveLoadMixin.h"
#include "network/include/SharedEmbeddingHolder.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufGameState = gamecore::serialization::ProtoBufGameState;
using ProtoBufGameInteraction = gamecore::serialization::ProtoBufGameInteraction;

struct GameEmbeddingImpl : torch::nn::Module, SaveLoadMixin<GameEmbeddingImpl> {
    GameEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    std::pair<torch::Tensor, torch::Tensor> embedGameState(const ProtoBufGameState& game_state);

    torch::Tensor embedGameInteraction(const std::vector<ProtoBufGameInteraction>& game_interactions,
                                       torch::Tensor card_indices, torch::Tensor cards);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    SharedEmbeddingHolder shared_embedding_holder_{nullptr};
    InstructionDataEmbedding instruction_data_embedding_{nullptr};
    InstructionEmbedding instruction_embedding_{nullptr};
    ConditionEmbedding condition_embedding_{nullptr};
    AttackEmbedding attack_embedding_{nullptr};
    AbilityEmbedding ability_embedding_{nullptr};
    GameStateEmbedding game_state_embedding_{nullptr};
    GameInteractionEmbedding game_interaction_embedding_{nullptr};
};

TORCH_MODULE(GameEmbedding);

#endif
