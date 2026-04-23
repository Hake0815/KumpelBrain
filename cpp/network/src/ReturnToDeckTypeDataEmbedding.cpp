#include "network/include/ReturnToDeckTypeDataEmbedding.h"

#include "network/include/SharedConstants.h"

using torch::indexing::Slice;

ReturnToDeckTypeDataEmbeddingImpl::ReturnToDeckTypeDataEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out, torch::Device device,
    torch::Dtype dtype) {
    card_position_embedding_ = shared_embedding_holder->card_position_embedding_;
    return_to_deck_type_embedding_ = register_module("return_to_deck_type_embedding",
                                                     torch::nn::Embedding(NUMBER_RETURN_TO_DECK_TYPES, dimension_out));

    to(device, dtype);
}

torch::Tensor ReturnToDeckTypeDataEmbeddingImpl::forward(const torch::Tensor& return_to_deck_type_data) {
    auto return_type = return_to_deck_type_data.index({Slice(), 0});
    auto position = return_to_deck_type_data.index({Slice(), 1});
    position =
        position + torch::ones_like(position, torch::TensorOptions().device(position.device()).dtype(position.dtype()));
    return return_to_deck_type_embedding_(return_type) + card_position_embedding_(position);
}
