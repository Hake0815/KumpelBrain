#ifndef CARD_AMOUNT_DATA_EMBEDDING_H
#define CARD_AMOUNT_DATA_EMBEDDING_H

#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include <torch/torch.h>

struct CardAmountDataEmbeddingImpl : torch::nn::Module,
                                     SaveLoadMixin<CardAmountDataEmbeddingImpl> {
  CardAmountDataEmbeddingImpl(
      std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
      int64_t dimension_out,
                              torch::Device device = torch::kCPU,
                              torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const torch::Tensor &card_amount_data);

private:
  NormalizedLinear card_amount_range_embedding_{nullptr};
  torch::nn::Embedding card_position_embedding_{nullptr};
};

TORCH_MODULE(CardAmountDataEmbedding);

#endif
