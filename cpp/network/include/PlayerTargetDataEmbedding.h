#ifndef PLAYER_TARGET_DATA_EMBEDDING_H
#define PLAYER_TARGET_DATA_EMBEDDING_H

#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include <torch/torch.h>

struct PlayerTargetDataEmbeddingImpl : torch::nn::Module,
                                       SaveLoadMixin<PlayerTargetDataEmbeddingImpl> {
  PlayerTargetDataEmbeddingImpl(
      std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
      int64_t dimension_out,
                                torch::Device device = torch::kCPU,
                                torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const torch::Tensor &player_target_data);

private:
  torch::nn::Embedding player_target_embedding_{nullptr};
};

TORCH_MODULE(PlayerTargetDataEmbedding);

#endif
