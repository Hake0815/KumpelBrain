#include "../include/ConditionEmbedding.h"

using torch::indexing::Slice;

ConditionEmbeddingImpl::ConditionEmbeddingImpl(
    std::shared_ptr<InstructionDataEmbeddingImpl> instruction_data_embedding,
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
    int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
  instruction_data_embedding_ =
      register_module("instruction_data_embedding", instruction_data_embedding);
  condition_type_embedding_ = register_module(
      "condition_type_embedding",
      torch::nn::Embedding(
          torch::nn::EmbeddingOptions(8, dimension_out_).padding_idx(0)));
  data_multi_head_attention_ = register_module(
      "data_multi_head_attention",
      MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                         std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0,
                         true, device_, dtype_));
  position_embedding_ = shared_embedding_holder->position_embedding_;
  conditions_multi_head_attention_ = register_module(
      "conditions_multi_head_attention",
      MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                         std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0,
                         true, device_, dtype_));
  to(device_, dtype_);
}

torch::Tensor ConditionEmbeddingImpl::forward(
    const std::vector<std::vector<gamecore::serialization::ProtoBufCondition>>
        &conditions_batch) {
  const int64_t batch_size = static_cast<int64_t>(conditions_batch.size());
  auto flat =
      nesting::flatten_conditions(conditions_batch, device_, torch::kInt64);

  auto data_tensors =
      compute_data_tensors(flat.instruction_indices, flat.instruction_data_types,
                           flat.instruction_data_type_indices,
                           flat.instruction_data, flat.filter_data,
                           flat.instruction_data_indices, batch_size);
  auto condition_embeddings = compute_condition_embeddings(
      flat.instruction_types, flat.instruction_indices,
      flat.instruction_data_type_indices, data_tensors);

  std::vector<torch::Tensor> batch_embeddings;
  batch_embeddings.reserve(batch_size);
  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    auto mask = flat.instruction_indices.index({Slice(), 0}).eq(batch_index);
    auto per_batch = condition_embeddings.index({mask});
    if (per_batch.size(0) == 0) {
      batch_embeddings.push_back(
          torch::zeros({dimension_out_},
                       torch::TensorOptions().device(device_).dtype(dtype_)));
      continue;
    }
    auto batched = position_embedding_(per_batch.unsqueeze(0)).squeeze(0);
    auto query = batched.unsqueeze(0);
    batch_embeddings.push_back(
        (query + conditions_multi_head_attention_(query, query, query))
            .sum(1)
            .squeeze(0));
  }

  if (batch_embeddings.empty()) {
    return torch::empty({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }
  return torch::stack(batch_embeddings, 0);
}

torch::Tensor ConditionEmbeddingImpl::compute_data_tensors(
    const torch::Tensor &condition_indices,
    const torch::Tensor &instruction_data_types,
    const torch::Tensor &instruction_data_type_indices,
    const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
    const std::vector<std::vector<gamecore::serialization::ProtoBufFilter>>
        &filter_data,
    const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
        &instruction_data_indices,
    int64_t batch_size) {
  return instruction_data_embedding_->forward(
      condition_indices, instruction_data_types, instruction_data_type_indices,
      instruction_data, filter_data, instruction_data_indices, batch_size);
}

torch::Tensor ConditionEmbeddingImpl::compute_condition_embeddings(
    const torch::Tensor &condition_types, const torch::Tensor &condition_indices,
    const torch::Tensor &instruction_data_type_indices,
    const torch::Tensor &data_tensors) {
  auto condition_type_embeddings =
      condition_type_embedding_(condition_types.to(torch::kLong));
  std::vector<torch::Tensor> condition_embeddings_list;
  const auto num_conditions = condition_indices.size(0);
  condition_embeddings_list.reserve(num_conditions);

  for (int64_t i = 0; i < num_conditions; ++i) {
    auto condition_index = condition_indices[i];
    auto mask = (instruction_data_type_indices.index({Slice(), Slice(0, 2)}) ==
                 condition_index)
                    .sum(1)
                    .eq(2);
    auto per_condition_data = data_tensors.index({mask});
    auto query = torch::cat(
                     {condition_type_embeddings[i].unsqueeze(0),
                      per_condition_data},
                     0)
                     .unsqueeze(0);
    auto embedded = (query + data_multi_head_attention_(query, query, query))
                        .sum(1)
                        .squeeze(0);
    condition_embeddings_list.push_back(embedded);
  }

  if (condition_embeddings_list.empty()) {
    return torch::empty({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }
  return torch::stack(condition_embeddings_list, 0);
}
