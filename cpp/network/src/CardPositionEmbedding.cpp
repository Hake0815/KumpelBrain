#include "../include/CardPositionEmbedding.h"

#include <vector>

namespace {

constexpr int64_t kNumOwners = 2;
constexpr int64_t kNumCardPositions = 11;
constexpr int64_t kNumPositionKnowledge = 3;

}  // namespace

CardPositionEmbeddingImpl::CardPositionEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    owner_embedding_ = register_module("owner_embedding",
                                       torch::nn::Embedding(torch::nn::EmbeddingOptions(kNumOwners, dimension_out)));
    possible_position_embedding_ = register_module(
        "possible_position_embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(kNumCardPositions + 1, dimension_out).padding_idx(0)));
    opponent_position_knowledge_embedding_ =
        register_module("opponent_position_knowledge_embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(kNumPositionKnowledge, dimension_out)));
    top_deck_position_index_embedding_ =
        register_module("top_deck_position_index_embedding", NormalizedLinear(1, dimension_out, 60.0, device, dtype));
    to(device, dtype);
}

torch::Tensor CardPositionEmbeddingImpl::forward(const std::vector<ProtoBufCardState>& card_state_batch) {
    const int64_t batch_size = card_state_batch.size();
    if (batch_size == 0) {
        return torch::empty({0, dimension_out_}, torch::TensorOptions().dtype(dtype_).device(device_));
    }
    auto index_options = torch::TensorOptions().dtype(torch::kInt64).device(device_);
    auto options = torch::TensorOptions().dtype(dtype_).device(device_);

    std::vector<int64_t> owner;
    owner.reserve(batch_size);
    std::vector<int64_t> possible_position_counts;
    possible_position_counts.reserve(batch_size);
    std::vector<int64_t> opponent_position_knowledge;
    opponent_position_knowledge.reserve(batch_size);
    std::vector<int64_t> top_deck_position_index;
    top_deck_position_index.reserve(batch_size);

    int max_possible_position_counts = 0;
    for (const auto& card_state : card_state_batch) {
        const auto& card_position = card_state.position();
        owner.push_back(static_cast<int64_t>(card_position.owner()));
        opponent_position_knowledge.push_back(static_cast<int64_t>(card_position.opponent_position_knowledge()));
        top_deck_position_index.push_back(static_cast<int64_t>(card_position.top_deck_position_index()));
        possible_position_counts.push_back(card_position.possible_positions_size());
        max_possible_position_counts = std::max(max_possible_position_counts, card_position.possible_positions_size());
    }

    std::vector<int64_t> possible_positions;
    possible_positions.reserve(batch_size * max_possible_position_counts);
    for (const auto& card_state : card_state_batch) {
        const auto& card_position = card_state.position();
        for (int i = 0; i < max_possible_position_counts; ++i) {
            if (i < card_position.possible_positions_size()) {
                possible_positions.push_back(card_position.possible_positions(i) + 1);
            } else {
                possible_positions.push_back(0);
            }
        }
    }
    const auto owner_tensor = torch::tensor(owner, index_options);
    const auto opponent_position_knowledge_tensor = torch::tensor(opponent_position_knowledge, index_options);
    const auto top_deck_position_index_tensor = torch::tensor(top_deck_position_index, options).unsqueeze(1);
    const auto possible_position_counts_tensor = torch::tensor(possible_position_counts, options);
    const auto possible_positions_tensor =
        torch::tensor(possible_positions, index_options).view({batch_size, max_possible_position_counts});

    const auto embedded_owner = owner_embedding_->forward(owner_tensor);
    const auto embedded_opponent_position_knowledge =
        opponent_position_knowledge_embedding_->forward(opponent_position_knowledge_tensor);
    const auto embedded_top_deck_position_index =
        top_deck_position_index_embedding_->forward(top_deck_position_index_tensor);
    const auto embedded_possible_positions = possible_position_embedding_->forward(possible_positions_tensor).sum(1);
    const auto embedded_possible_positions_averaged =
        embedded_possible_positions / possible_position_counts_tensor.unsqueeze(1).clamp_min(1.0f);

    return (embedded_owner + embedded_opponent_position_knowledge + embedded_top_deck_position_index +
            embedded_possible_positions_averaged) /
           4.0;
}
