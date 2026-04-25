#ifndef PLAYER_STATE_EMBEDDING_H
#define PLAYER_STATE_EMBEDDING_H

#include <torch/nn/modules/embedding.h>
#include <torch/torch.h>

#include "network/include/MultiHeadAttention.h"
#include "network/include/NormalizedLinear.h"
#include "network/include/SaveLoadMixin.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufPlayerState = gamecore::serialization::ProtoBufPlayerState;

struct PlayerStateEmbeddingImpl : torch::nn::Module, SaveLoadMixin<PlayerStateEmbeddingImpl> {
    PlayerStateEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                             torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const ProtoBufPlayerState& self_player_state,
                          const ProtoBufPlayerState& opponent_player_state);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::TensorOptions index_options_;
    torch::TensorOptions float_options_;
    torch::nn::Embedding is_active_embedding_{nullptr};
    torch::nn::Embedding is_attacking_embedding_{nullptr};
    torch::nn::Embedding knows_his_prizes_embedding_{nullptr};
    NormalizedLinear hand_count_embedding_{nullptr};
    NormalizedLinear deck_count_embedding_{nullptr};
    NormalizedLinear prizes_count_embedding_{nullptr};
    NormalizedLinear bench_count_embedding_{nullptr};
    NormalizedLinear discard_pile_count_embedding_{nullptr};
    NormalizedLinear turn_counter_embedding_{nullptr};
    torch::nn::Embedding player_turn_traits_embedding_{nullptr};

    torch::nn::Embedding queries_embedding_{nullptr};
    MultiHeadAttention multi_head_attention_{nullptr};

    torch::Tensor embed_player_state(const ProtoBufPlayerState& player_state);
};

TORCH_MODULE(PlayerStateEmbedding);

#endif
