#include "network/include/GameStateEmbedding.h"

#include "network/include/GameStateBatch.h"

namespace {
constexpr int64_t kNumPlayersPerGame = 2;
}  // namespace

GameStateEmbeddingImpl::GameStateEmbeddingImpl(std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
                                               int64_t dimension_out,
                                               const SharedInstructionEmbeddings& shared_instruction_embeddings,
                                               torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    player_state_embedding_ =
        register_module("player_state_embedding", PlayerStateEmbedding(dimension_out, device, dtype));
    card_state_embedding_ = register_module(
        "card_state_embedding",
        CardStateEmbedding(shared_embedding_holder, dimension_out, shared_instruction_embeddings, device, dtype));
    to(device, dtype);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GameStateEmbeddingImpl::forward(
    const std::vector<ProtoBufGameState>& game_states) {
    const int64_t batch_size = static_cast<int64_t>(game_states.size());
    auto options = torch::TensorOptions().device(device_).dtype(dtype_);
    auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);

    if (batch_size == 0) {
        return {torch::empty({0, 0, dimension_out_}, options), torch::empty({0, 0}, mask_options),
                torch::empty({0, 0}, torch::TensorOptions().device(device_).dtype(torch::kInt64))};
    }

    std::vector<ProtoBufPlayerState> self_states;
    std::vector<ProtoBufPlayerState> opponent_states;
    self_states.reserve(static_cast<size_t>(batch_size));
    opponent_states.reserve(static_cast<size_t>(batch_size));
    for (const auto& game_state : game_states) {
        self_states.push_back(game_state.self_state());
        opponent_states.push_back(game_state.opponent_state());
    }

    google::protobuf::RepeatedPtrField<ProtoBufCardState> all_cards;
    CardSegmentInfo segments;
    append_card_states(game_states, all_cards, segments);

    auto player_states = player_state_embedding_(self_states, opponent_states);
    auto [card_states, card_mask, card_indices] =
        card_state_embedding_(all_cards, segments.offsets);

    const int64_t max_cards = card_states.size(1);
    const int64_t total_rows = kNumPlayersPerGame + max_cards;
    auto embedding = torch::zeros({batch_size, total_rows, dimension_out_}, options);
    auto mask = torch::zeros({batch_size, total_rows}, mask_options);

    embedding.slice(1, 0, kNumPlayersPerGame) = player_states;
    mask.slice(1, 0, kNumPlayersPerGame).fill_(true);
    if (max_cards > 0) {
        embedding.slice(1, kNumPlayersPerGame, total_rows) = card_states;
        mask.slice(1, kNumPlayersPerGame, total_rows) = card_mask;
    }

    return {embedding, mask, card_indices};
}
