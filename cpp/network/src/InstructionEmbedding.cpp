#include "../include/InstructionEmbedding.h"

using torch::indexing::Slice;

InstructionEmbeddingImpl::InstructionEmbeddingImpl(
    std::shared_ptr<InstructionDataEmbeddingImpl> instruction_data_embedding,
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
    int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
  instruction_data_embedding_ =
      register_module("instruction_data_embedding", instruction_data_embedding);
  instruction_type_embedding_ = register_module(
      "instruction_type_embedding",
      torch::nn::Embedding(
          torch::nn::EmbeddingOptions(8, dimension_out_).padding_idx(0)));
  data_multi_head_attention_ = register_module(
      "data_multi_head_attention",
      MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                         std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0,
                         true, device_, dtype_));
  position_embedding_ = shared_embedding_holder->position_embedding_;
  instructions_multi_head_attention_ = register_module(
      "instructions_multi_head_attention",
      MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                         std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0,
                         true, device_, dtype_));
  to(device_, dtype_);
}

torch::Tensor InstructionEmbeddingImpl::forward(
    const std::vector<std::vector<gamecore::serialization::ProtoBufInstruction>>
        &instructions_batch) {
  const int64_t batch_size = static_cast<int64_t>(instructions_batch.size());

  auto flat = nesting::flatten_instructions(instructions_batch, device_,
                                            torch::kInt64);
  return forward_flattened(flat.instruction_types, flat.instruction_indices,
                           flat.instruction_data_types,
                           flat.instruction_data_type_indices,
                           flat.instruction_data, flat.filter_data,
                           flat.instruction_data_indices, batch_size);
}

torch::Tensor InstructionEmbeddingImpl::forward_flattened(
    const torch::Tensor &instruction_types,
    const torch::Tensor &instruction_indices,
    const torch::Tensor &instruction_data_types,
    const torch::Tensor &instruction_data_type_indices,
    const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
    const std::vector<std::vector<gamecore::serialization::ProtoBufFilter>>
        &filter_data,
    const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
        &instruction_data_indices,
    int64_t batch_size) {
  auto data_tensors =
      compute_data_tensors(instruction_indices, instruction_data_types,
                           instruction_data_type_indices, instruction_data,
                           filter_data, instruction_data_indices, batch_size);
  auto instruction_embeddings = compute_instruction_embeddings(
      instruction_types, instruction_indices, instruction_data_type_indices,
      data_tensors);

  std::vector<torch::Tensor> batch_embeddings;
  batch_embeddings.reserve(batch_size);
  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    auto mask = instruction_indices.index({Slice(), 0}).eq(batch_index);
    auto per_batch = instruction_embeddings.index({mask});
    if (per_batch.size(0) == 0) {
      batch_embeddings.push_back(
          torch::zeros({dimension_out_},
                       torch::TensorOptions().device(device_).dtype(dtype_)));
      continue;
    }
    auto batched = position_embedding_(per_batch.unsqueeze(0)).squeeze(0);
    auto query = batched.unsqueeze(0);
    batch_embeddings.push_back(
        (query + instructions_multi_head_attention_(query, query, query))
            .sum(1)
            .squeeze(0));
  }

  if (batch_embeddings.empty()) {
    return torch::empty({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }
  return torch::stack(batch_embeddings, 0);
}

torch::Tensor InstructionEmbeddingImpl::compute_data_tensors(
    const torch::Tensor &instruction_indices,
    const torch::Tensor &instruction_data_types,
    const torch::Tensor &instruction_data_type_indices,
    const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
    const std::vector<std::vector<gamecore::serialization::ProtoBufFilter>>
        &filter_data,
    const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
        &instruction_data_indices,
    int64_t batch_size) {
  return instruction_data_embedding_->forward(
      instruction_indices, instruction_data_types,
      instruction_data_type_indices, instruction_data, filter_data,
      instruction_data_indices, batch_size);
}

torch::Tensor InstructionEmbeddingImpl::compute_instruction_embeddings(
    const torch::Tensor &instruction_types,
    const torch::Tensor &instruction_indices,
    const torch::Tensor &instruction_data_type_indices,
    const torch::Tensor &data_tensors) {
  auto instruction_type_embeddings =
      instruction_type_embedding_(instruction_types.to(torch::kLong));
  std::vector<torch::Tensor> instruction_embeddings_list;
  const auto num_instructions = instruction_indices.size(0);
  instruction_embeddings_list.reserve(num_instructions);
  for (int64_t i = 0; i < num_instructions; ++i) {
    auto instruction_index = instruction_indices[i];
    auto mask = (instruction_data_type_indices.index({Slice(), Slice(0, 2)}) ==
                 instruction_index)
                    .sum(1)
                    .eq(2);
    auto per_instruction_data = data_tensors.index({mask});
    auto query = torch::cat({instruction_type_embeddings[i].unsqueeze(0),
                             per_instruction_data},
                            0)
                     .unsqueeze(0);
    auto embedded = (query + data_multi_head_attention_(query, query, query))
                        .sum(1)
                        .squeeze(0);
    instruction_embeddings_list.push_back(embedded);
  }
  if (instruction_embeddings_list.empty()) {
    return torch::empty({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }
  return torch::stack(instruction_embeddings_list, 0);
}
