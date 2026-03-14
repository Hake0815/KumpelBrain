#include "../include/PlayerTargetDataEmbedding.h"

PlayerTargetDataEmbeddingImpl::PlayerTargetDataEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
    int64_t dimension_out, torch::Device device, torch::Dtype dtype) {
  (void)dimension_out;
  (void)device;
  (void)dtype;
  player_target_embedding_ = shared_embedding_holder->player_target_embedding_;
}

torch::Tensor PlayerTargetDataEmbeddingImpl::forward(
    const torch::Tensor &player_target_data) {
  return player_target_embedding_(player_target_data);
}
