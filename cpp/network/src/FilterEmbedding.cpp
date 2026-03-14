#include "../include/FilterEmbedding.h"
#include "../include/TensorUtils.h"
#include <algorithm>

using torch::indexing::Slice;

FilterEmbeddingImpl::FilterEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
    int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
  logical_operator_embedding_ = register_module(
      "logical_operator_embedding",
      torch::nn::Embedding(
          torch::nn::EmbeddingOptions(3, dimension_out_).padding_idx(0)));
  filter_condition_embedding_ = register_module(
      "filter_condition_embedding",
      FilterConditionEmbedding(shared_embedding_holder, dimension_out_, device_,
                               dtype_));
  multi_head_attention_ = register_module(
      "multi_head_attention",
      MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                         std::max<int64_t>(dimension_out_ / 16, 1), 2, 0.0,
                         false, device_, dtype_));
  to(device_, dtype_);
}

torch::Tensor FilterEmbeddingImpl::combine_condition(
    const std::vector<torch::Tensor> &filter_conditions,
    std::optional<int64_t> op) {
  if (filter_conditions.size() == 1) {
    return filter_conditions[0];
  }

  const int64_t operator_value = op.value_or(0);
  if (operator_tensor_cache_.find(operator_value) ==
      operator_tensor_cache_.end()) {
    operator_tensor_cache_[operator_value] = torch::tensor(
        operator_value,
        torch::TensorOptions().device(device_).dtype(torch::kLong));
  }

  auto embedded_operator =
      logical_operator_embedding_(operator_tensor_cache_[operator_value])
          .unsqueeze(0);
  auto filter_conditions_stacked = torch::stack(filter_conditions, 0);
  auto query = torch::cat({filter_conditions_stacked, embedded_operator}, 0)
                   .unsqueeze(0);
  auto updated_query =
      (multi_head_attention_(query, query, query) + query).squeeze(0);
  return updated_query.sum(0);
}

torch::Tensor
FilterEmbeddingImpl::forward(
    const std::vector<gamecore::serialization::ProtoBufFilter> &filter) {
  if (filter.empty()) {
    return torch::zeros({dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }

  auto traverse_entries = nesting::traverse_filter(filter);
  auto flat = nesting::flatten(traverse_entries);

  std::vector<std::vector<int64_t>> flattened = flat.flattened_input;
  auto flattened_tensor =
      tensor_utils::tensor_from_2d_int64(flattened, device_, torch::kLong);
  auto field_type = flattened_tensor.index({Slice(), 0});
  auto comparison_operator = flattened_tensor.index({Slice(), 1});
  auto value = flattened_tensor.index({Slice(), 2});

  auto embedded_conditions = filter_condition_embedding_->forward(
      field_type, comparison_operator, value);

  std::vector<torch::Tensor> embedded_condition_list;
  embedded_condition_list.reserve(embedded_conditions.size(0));
  for (int64_t i = 0; i < embedded_conditions.size(0); ++i) {
    embedded_condition_list.push_back(embedded_conditions[i]);
  }

  auto reduced =
      nesting::reduce(embedded_condition_list, flat.groups, flat.operators,
                      [this](const std::vector<torch::Tensor> &conditions,
                             std::optional<int64_t> op) {
                        return combine_condition(conditions, op);
                      });

  if (reduced.empty()) {
    return torch::zeros({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }
  return torch::stack(reduced, 0);
}
