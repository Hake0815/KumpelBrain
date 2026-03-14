#ifndef POSITIONAL_EMBEDDING_H
#define POSITIONAL_EMBEDDING_H

#include <torch/torch.h>

struct PositionalEmbeddingImpl : torch::nn::Module {
  PositionalEmbeddingImpl(int64_t d_model, double dropout = 0.1,
                          int64_t max_len = 5000,
                          torch::Device device = torch::kCPU,
                          torch::Dtype dtype = torch::kFloat);
  torch::Tensor forward(const torch::Tensor &x);

private:
  torch::nn::Dropout dropout_{nullptr};
  torch::Tensor positional_embedding_;
};

TORCH_MODULE(PositionalEmbedding);

#endif