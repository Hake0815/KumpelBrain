#ifndef DISCARD_DATA_EMBEDDING_H
#define DISCARD_DATA_EMBEDDING_H

#include "../include/SaveLoadMixin.h"
#include <torch/torch.h>

struct DiscardDataEmbeddingImpl : torch::nn::Module,
                                  SaveLoadMixin<DiscardDataEmbeddingImpl> {
  DiscardDataEmbeddingImpl(int64_t dimension_out,
                           torch::Device device = torch::kCPU,
                           torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const torch::Tensor &discard_data);

private:
  torch::nn::Embedding target_source_embedding_{nullptr};
};

TORCH_MODULE(DiscardDataEmbedding);

#endif
