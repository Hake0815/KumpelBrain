#ifndef FILTER_CONDITION_EMBEDDING_H
#define FILTER_CONDITION_EMBEDDING_H

#include "../include/MultiHeadAttention.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include <torch/torch.h>

struct FilterConditionEmbeddingImpl
    : torch::nn::Module,
      SaveLoadMixin<FilterConditionEmbeddingImpl> {
  FilterConditionEmbeddingImpl(
      std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
      int64_t dimension_out, torch::Device device = torch::kCPU,
      torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const torch::Tensor &field_type,
                        const torch::Tensor &comparison_operator,
                        const torch::Tensor &value);

private:
  torch::nn::Embedding card_type_embedding_{nullptr};
  torch::nn::Embedding card_subtype_embedding_{nullptr};
  NormalizedLinear hp_embedding_{nullptr};
  int64_t dimension_out_;
  torch::Device device_;
  torch::Dtype dtype_;
  torch::nn::Embedding filter_field_embedding_{nullptr};
  torch::nn::Embedding filter_operation_embedding_{nullptr};
  MultiHeadAttention multi_head_attention_{nullptr};
};

TORCH_MODULE(FilterConditionEmbedding);

#endif
