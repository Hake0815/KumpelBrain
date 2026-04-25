#include "network/include/GameStateEmbedding.h"

GameStateEmbeddingImpl::GameStateEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    player_state_embedding_ =
        register_module("player_state_embedding", PlayerStateEmbedding(dimension_out, device, dtype));
    card_state_embedding_ = register_module("card_state_embedding", CardStateEmbedding(dimension_out, device, dtype));
    to(device, dtype);
}

torch::Tensor GameStateEmbeddingImpl::forward(const ProtoBufGameState& game_state) {
    auto player_states = player_state_embedding_(game_state.self_state(), game_state.opponent_state());
    auto card_states = card_state_embedding_(game_state.card_states());
    return torch::cat({player_states, card_states}, 0);
}