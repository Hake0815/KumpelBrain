#include "../include/InstructionEmbedding.h"
#include "../include/AttentionUtils.h"
#include "../include/Nesting.h"
#include "../include/TensorUtils.h"
#include <algorithm>

namespace serialization = gamecore::serialization;

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
    const std::vector<std::vector<serialization::ProtoBufInstruction>>
        &instructions_batch) {
  const int64_t batch_size = static_cast<int64_t>(instructions_batch.size());

  auto flat = nesting::flatten_instructions(instructions_batch, device_,
                                            torch::kInt64);
  return forward_flattened(flat.instruction_types, flat.instruction_indices,
                           flat.instruction_data_types,
                           flat.instruction_data_parent_rows,
                           flat.instruction_data_type_indices,
                           flat.instruction_data, flat.filter_data,
                           flat.instruction_data_indices, batch_size);
}

torch::Tensor InstructionEmbeddingImpl::forward_flattened(
    const torch::Tensor &instruction_types,
    const torch::Tensor &instruction_indices,
    const torch::Tensor &instruction_data_types,
    const torch::Tensor &instruction_data_parent_rows,
    const torch::Tensor &instruction_data_type_indices,
    const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
    const std::vector<std::vector<serialization::ProtoBufFilter>> &filter_data,
    const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
        &instruction_data_indices,
    int64_t batch_size) {
  auto data_tensors =
      compute_data_tensors(instruction_indices, instruction_data_types,
                           instruction_data_type_indices, instruction_data,
                           filter_data, instruction_data_indices, batch_size);
  auto instruction_embeddings = compute_instruction_embeddings(
      instruction_types, instruction_indices, instruction_data_parent_rows,
      data_tensors);
  auto batch_offsets = tensor_utils::build_contiguous_offsets(
      instruction_indices.select(1, 0), batch_size);
  auto [padded_batch, valid_token_mask] =
      tensor_utils::pad_by_offsets(instruction_embeddings, batch_offsets,
                                   dimension_out_);
  auto positioned = position_embedding_(padded_batch);
  return attention_utils::masked_self_attention_reduce(
      instructions_multi_head_attention_, positioned, valid_token_mask);
}

torch::Tensor InstructionEmbeddingImpl::compute_data_tensors(
    const torch::Tensor &instruction_indices,
    const torch::Tensor &instruction_data_types,
    const torch::Tensor &instruction_data_type_indices,
    const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
    const std::vector<std::vector<serialization::ProtoBufFilter>> &filter_data,
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
    const torch::Tensor &instruction_data_parent_rows,
    const torch::Tensor &data_tensors) {
  auto instruction_type_embeddings =
      instruction_type_embedding_(instruction_types.to(torch::kLong));
  const auto num_instructions = instruction_indices.size(0);
  if (num_instructions == 0) {
    return torch::empty({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }

  auto data_offsets = tensor_utils::build_parent_offsets(
      instruction_data_parent_rows, num_instructions);
  auto [padded_data, data_token_mask] =
      tensor_utils::pad_by_offsets(data_tensors, data_offsets, dimension_out_);
  auto queries = torch::cat({instruction_type_embeddings.unsqueeze(1), padded_data},
                            1);
  auto valid_token_mask = torch::cat(
      {torch::ones({num_instructions, 1},
                   torch::TensorOptions().device(device_).dtype(torch::kBool)),
       data_token_mask},
      1);
  return attention_utils::masked_self_attention_reduce(
      data_multi_head_attention_, queries, valid_token_mask);
}
