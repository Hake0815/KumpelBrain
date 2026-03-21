#include "../include/FilterConditionEmbedding.h"
#include <algorithm>

FilterConditionEmbeddingImpl::FilterConditionEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
    int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
  card_type_embedding_ = shared_embedding_holder->card_type_embedding_;
  card_subtype_embedding_ = shared_embedding_holder->card_subtype_embedding_;
  hp_embedding_ = shared_embedding_holder->hp_embedding_;
  filter_field_embedding_ = register_module(
      "filter_field_embedding",
      torch::nn::Embedding(
          torch::nn::EmbeddingOptions(6, dimension_out_).padding_idx(0)));
  filter_operation_embedding_ = register_module(
      "filter_operation_embedding",
      torch::nn::Embedding(
          torch::nn::EmbeddingOptions(5, dimension_out_).padding_idx(0)));
  multi_head_attention_ = register_module(
      "multi_head_attention",
      MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                         std::max<int64_t>(dimension_out_ / 16, 1), 2, 0.0,
                         false, device_, dtype_));

  to(device_, dtype_);
}

torch::Tensor
FilterConditionEmbeddingImpl::forward(const torch::Tensor &field_type,
                                      const torch::Tensor &comparison_operator,
                                      const torch::Tensor &value) {
  auto field_embedding = filter_field_embedding_(field_type.to(torch::kLong));
  auto operation_embedding =
      filter_operation_embedding_(comparison_operator.to(torch::kLong));

  auto value_embedding =
      torch::zeros({field_type.size(0), dimension_out_},
                   torch::TensorOptions().device(device_).dtype(dtype_));

  auto mask_3 = field_type == 3;
  auto mask_4 = field_type == 4;
  auto mask_5 = field_type == 5;

  auto idx_3 = torch::nonzero(mask_3).squeeze(1);
  if (idx_3.numel() > 0) {
    value_embedding.index_put_(
        {idx_3}, card_type_embedding_(value.index({idx_3}).to(torch::kLong)));
  }
  auto idx_4 = torch::nonzero(mask_4).squeeze(1);
  if (idx_4.numel() > 0) {
    value_embedding.index_put_(
        {idx_4},
        card_subtype_embedding_(value.index({idx_4}).to(torch::kLong)));
  }
  auto idx_5 = torch::nonzero(mask_5).squeeze(1);
  if (idx_5.numel() > 0) {
    value_embedding.index_put_(
        {idx_5}, hp_embedding_(value.index({idx_5}).to(dtype_).unsqueeze(1)));
  }

  auto query =
      torch::stack({field_embedding, operation_embedding, value_embedding}, 1);
  auto updated_query = multi_head_attention_(query, query, query) + query;
  return torch::sum(updated_query, 1);
}
