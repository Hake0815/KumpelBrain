#ifndef FILTER_EMBEDDING_H
#define FILTER_EMBEDDING_H

#include "../include/FilterConditionEmbedding.h"
#include "../include/MultiHeadAttention.h"
#include "../include/Nesting.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include <torch/torch.h>
#include <unordered_map>

struct FilterEmbeddingImpl : torch::nn::Module, SaveLoadMixin<FilterEmbeddingImpl> {
  FilterEmbeddingImpl(
      std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
      int64_t dimension_out,
                      torch::Device device = torch::kCPU,
                      torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const std::vector<nesting::FilterNode> &filter);

private:
  torch::Tensor combine_condition(const std::vector<torch::Tensor> &filter_conditions,
                                  std::optional<int64_t> op);

  int64_t dimension_out_;
  torch::Device device_;
  torch::Dtype dtype_;
  torch::nn::Embedding logical_operator_embedding_{nullptr};
  FilterConditionEmbedding filter_condition_embedding_{nullptr};
  MultiHeadAttention multi_head_attention_{nullptr};
  std::unordered_map<int64_t, torch::Tensor> operator_tensor_cache_{};
};

TORCH_MODULE(FilterEmbedding);

#endif
