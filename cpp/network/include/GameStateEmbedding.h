#ifndef GAME_STATE_EMBEDDING_H
#define GAME_STATE_EMBEDDING_H

#include <torch/torch.h>

#include "network/include/CardStateEmbedding.h"
#include "network/include/PlayerStateEmbedding.h"
#include "network/include/SaveLoadMixin.h"
#include "network/include/SharedInstructionEmbeddings.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufGameState = gamecore::serialization::ProtoBufGameState;

struct GameStateEmbeddingImpl : torch::nn::Module, SaveLoadMixin<GameStateEmbeddingImpl> {
    GameStateEmbeddingImpl(std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out,
                           const SharedInstructionEmbeddings& shared_instruction_embeddings,
                           torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const std::vector<ProtoBufGameState>& game_states);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    PlayerStateEmbedding player_state_embedding_{nullptr};
    CardStateEmbedding card_state_embedding_{nullptr};
};

TORCH_MODULE(GameStateEmbedding);

#endif
