#include "../include/CardAmountDataEmbedding.h"

using torch::indexing::Slice;

CardAmountDataEmbeddingImpl::CardAmountDataEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out, torch::Device device,
    torch::Dtype dtype) {
    (void)dimension_out;
    (void)device;
    (void)dtype;

    card_amount_range_embedding_ = shared_embedding_holder->card_amount_range_embedding_;
    card_position_embedding_ = shared_embedding_holder->card_position_embedding_;
}

torch::Tensor CardAmountDataEmbeddingImpl::forward(const torch::Tensor& card_amount_data) {
    auto amount_range = card_amount_data.index({Slice(), Slice(0, 2)});
    auto position = card_amount_data.index({Slice(), 2});
    return card_amount_range_embedding_(amount_range) + card_position_embedding_(position);
}
