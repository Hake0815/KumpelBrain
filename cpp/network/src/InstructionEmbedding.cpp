#include "../include/InstructionEmbedding.h"
#include "../include/Nesting.h"
#include "../include/TensorUtils.h"
#include <algorithm>

namespace serialization = gamecore::serialization;

using torch::indexing::Slice;

namespace {

torch::Tensor masked_self_attention_reduce(
    MultiHeadAttention &multi_head_attention, const torch::Tensor &padded_sequences,
    const torch::Tensor &valid_token_mask) {
  if (padded_sequences.size(1) == 0) {
    return torch::zeros(
        {padded_sequences.size(0), padded_sequences.size(2)},
        torch::TensorOptions()
            .device(padded_sequences.device())
            .dtype(padded_sequences.dtype()));
  }

  auto attention_mask = tensor_utils::make_padding_attention_mask(
      valid_token_mask, padded_sequences.scalar_type());
  auto attended = padded_sequences + multi_head_attention(
                                       padded_sequences, padded_sequences,
                                       padded_sequences, attention_mask);
  return tensor_utils::masked_sequence_sum(attended, valid_token_mask);
}

torch::Tensor build_grouped_queries(const torch::Tensor &type_embeddings,
                                    const std::vector<int64_t> &data_offsets,
                                    const torch::Tensor &data_tensors) {
  const auto num_groups = type_embeddings.size(0);
  if (num_groups == 0) {
    return torch::empty({0, 0, type_embeddings.size(1)},
                        torch::TensorOptions()
                            .device(type_embeddings.device())
                            .dtype(type_embeddings.dtype()));
  }

  int64_t max_data_length = 0;
  for (int64_t group_index = 0; group_index < num_groups; ++group_index) {
    max_data_length = std::max(
        max_data_length, data_offsets[static_cast<size_t>(group_index + 1)] -
                             data_offsets[static_cast<size_t>(group_index)]);
  }

  auto queries = torch::zeros(
      {num_groups, max_data_length + 1, type_embeddings.size(1)},
      torch::TensorOptions()
          .device(type_embeddings.device())
          .dtype(type_embeddings.dtype()));
  queries.index_put_({Slice(), 0}, type_embeddings);

  for (int64_t group_index = 0; group_index < num_groups; ++group_index) {
    const auto start = data_offsets[static_cast<size_t>(group_index)];
    const auto end = data_offsets[static_cast<size_t>(group_index + 1)];
    const auto length = end - start;
    if (length == 0) {
      continue;
    }

    queries.index_put_({group_index, Slice(1, 1 + length)},
                       data_tensors.index({Slice(start, end)}));
  }

  return queries;
}

torch::Tensor build_grouped_query_mask(const std::vector<int64_t> &data_offsets,
                                       torch::Device device) {
  const auto num_groups =
      static_cast<int64_t>(data_offsets.empty() ? 0 : data_offsets.size() - 1);
  if (num_groups == 0) {
    return torch::empty({0, 0},
                        torch::TensorOptions().device(device).dtype(torch::kBool));
  }

  int64_t max_data_length = 0;
  for (int64_t group_index = 0; group_index < num_groups; ++group_index) {
    max_data_length = std::max(
        max_data_length, data_offsets[static_cast<size_t>(group_index + 1)] -
                             data_offsets[static_cast<size_t>(group_index)]);
  }

  auto valid_token_mask = torch::zeros(
      {num_groups, max_data_length + 1},
      torch::TensorOptions().device(device).dtype(torch::kBool));
  valid_token_mask.index_put_({Slice(), 0}, true);

  for (int64_t group_index = 0; group_index < num_groups; ++group_index) {
    const auto start = data_offsets[static_cast<size_t>(group_index)];
    const auto end = data_offsets[static_cast<size_t>(group_index + 1)];
    const auto length = end - start;
    if (length == 0) {
      continue;
    }

    valid_token_mask.index_put_({group_index, Slice(1, 1 + length)}, true);
  }

  return valid_token_mask;
}

} // namespace

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
    const std::vector<std::vector<serialization::ProtoBufFilter>> &filter_data,
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
  auto batch_offsets = tensor_utils::build_contiguous_offsets(
      instruction_indices.index({Slice(), 0}), batch_size);
  auto [padded_batch, valid_token_mask] =
      tensor_utils::pad_by_offsets(instruction_embeddings, batch_offsets,
                                   dimension_out_);
  auto positioned = position_embedding_(padded_batch);
  return masked_self_attention_reduce(instructions_multi_head_attention_,
                                      positioned, valid_token_mask);
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
    const torch::Tensor &instruction_data_type_indices,
    const torch::Tensor &data_tensors) {
  auto instruction_type_embeddings =
      instruction_type_embedding_(instruction_types.to(torch::kLong));
  const auto num_instructions = instruction_indices.size(0);
  if (num_instructions == 0) {
    return torch::empty({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }

  auto data_offsets = tensor_utils::build_parent_offsets(
      instruction_indices, instruction_data_type_indices, 2);
  auto queries =
      build_grouped_queries(instruction_type_embeddings, data_offsets, data_tensors);
  auto valid_token_mask = build_grouped_query_mask(data_offsets, device_);
  return masked_self_attention_reduce(data_multi_head_attention_, queries,
                                      valid_token_mask);
}
