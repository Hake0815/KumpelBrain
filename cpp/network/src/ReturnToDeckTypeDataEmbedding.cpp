#include "../include/ReturnToDeckTypeDataEmbedding.h"

using torch::indexing::Slice;

ReturnToDeckTypeDataEmbeddingImpl::ReturnToDeckTypeDataEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
    int64_t dimension_out, torch::Device device, torch::Dtype dtype) {
  card_position_embedding_ = shared_embedding_holder->card_position_embedding_;
  return_to_deck_type_embedding_ = register_module(
      "return_to_deck_type_embedding", torch::nn::Embedding(2, dimension_out));

  to(device, dtype);
}

torch::Tensor ReturnToDeckTypeDataEmbeddingImpl::forward(
    const torch::Tensor &return_to_deck_type_data) {
  auto return_type = return_to_deck_type_data.index({Slice(), 0});
  auto position = return_to_deck_type_data.index({Slice(), 1});
  return return_to_deck_type_embedding_(return_type) +
         card_position_embedding_(position);
}
