#ifndef RETURN_TO_DECK_TYPE_DATA_EMBEDDING_H
#define RETURN_TO_DECK_TYPE_DATA_EMBEDDING_H

#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include <torch/torch.h>

struct ReturnToDeckTypeDataEmbeddingImpl
    : torch::nn::Module, SaveLoadMixin<ReturnToDeckTypeDataEmbeddingImpl> {
  ReturnToDeckTypeDataEmbeddingImpl(SharedEmbeddingHolder shared_embedding_holder,
                                    int64_t dimension_out,
                                    torch::Device device = torch::kCPU,
                                    torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const torch::Tensor &return_to_deck_type_data);

private:
  torch::nn::Embedding card_position_embedding_{nullptr};
  torch::nn::Embedding return_to_deck_type_embedding_{nullptr};
};

TORCH_MODULE(ReturnToDeckTypeDataEmbedding);

#endif
