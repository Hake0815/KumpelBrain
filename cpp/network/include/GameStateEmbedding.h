#ifndef GAME_STATE_EMBEDDING_H
#define GAME_STATE_EMBEDDING_H

#include <torch/torch.h>

#include "network/include/CardStateEmbedding.h"
#include "network/include/PlayerStateEmbedding.h"
#include "network/include/SaveLoadMixin.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufGameState = gamecore::serialization::ProtoBufGameState;

struct GameStateEmbeddingImpl : torch::nn::Module, SaveLoadMixin<GameStateEmbeddingImpl> {
    GameStateEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                           torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const ProtoBufGameState& game_state);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    PlayerStateEmbedding player_state_embedding_{nullptr};
    CardStateEmbedding card_state_embedding_{nullptr};
};

TORCH_MODULE(GameStateEmbedding);

#endif
