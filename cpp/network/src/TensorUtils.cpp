#include "../include/TensorUtils.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tensor_utils {

namespace {

torch::Tensor to_cpu_long(const torch::Tensor &tensor) {
  return tensor.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong));
}

bool row_prefix_matches(const torch::TensorAccessor<int64_t, 2> &lhs,
                        int64_t lhs_row,
                        const torch::TensorAccessor<int64_t, 2> &rhs,
                        int64_t rhs_row, int64_t columns) {
  for (int64_t column = 0; column < columns; ++column) {
    if (lhs[lhs_row][column] != rhs[rhs_row][column]) {
      return false;
    }
  }
  return true;
}

} // namespace

torch::Tensor
tensor_from_2d_int64(const std::vector<std::vector<int64_t>> &values,
                     std::optional<torch::Device> device,
                     std::optional<torch::Dtype> dtype) {
  auto options = torch::TensorOptions();
  if (device.has_value()) {
    options = options.device(*device);
  }
  if (dtype.has_value()) {
    options = options.dtype(*dtype);
  } else {
    options = options.dtype(torch::kInt64);
  }

  if (values.empty()) {
    return torch::empty({0, 0}, options);
  }

  const auto width = static_cast<int64_t>(values.front().size());
  std::vector<int64_t> flat;
  flat.reserve(values.size() * static_cast<size_t>(width));

  for (const auto &row : values) {
    if (static_cast<int64_t>(row.size()) != width) {
      throw std::invalid_argument("Inconsistent row size in 2D tensor input");
    }
    flat.insert(flat.end(), row.begin(), row.end());
  }

  return torch::tensor(flat, options)
      .view({static_cast<int64_t>(values.size()), width});
}

std::vector<int64_t> build_contiguous_offsets(const torch::Tensor &group_indices,
                                              int64_t num_groups) {
  std::vector<int64_t> offsets(static_cast<size_t>(num_groups) + 1, 0);
  if (num_groups <= 0) {
    return offsets;
  }

  const auto group_indices_cpu = to_cpu_long(group_indices.contiguous()).view({-1});
  const auto accessor = group_indices_cpu.accessor<int64_t, 1>();
  int64_t cursor = 0;

  for (int64_t group = 0; group < num_groups; ++group) {
    offsets[static_cast<size_t>(group)] = cursor;
    while (cursor < group_indices_cpu.size(0) && accessor[cursor] == group) {
      ++cursor;
    }
  }

  offsets[static_cast<size_t>(num_groups)] = group_indices_cpu.size(0);
  return offsets;
}

std::vector<int64_t>
build_parent_offsets(const torch::Tensor &parent_indices,
                     const torch::Tensor &child_indices, int64_t parent_columns) {
  const auto parent_count = parent_indices.size(0);
  std::vector<int64_t> offsets(static_cast<size_t>(parent_count) + 1, 0);
  if (parent_count == 0) {
    return offsets;
  }

  const auto child_count = child_indices.size(0);
  if (child_count == 0) {
    return offsets;
  }

  const auto parent_cpu = to_cpu_long(parent_indices.contiguous());
  const auto child_cpu = to_cpu_long(child_indices.contiguous());
  const auto parent_accessor = parent_cpu.accessor<int64_t, 2>();
  const auto child_accessor = child_cpu.accessor<int64_t, 2>();

  int64_t child_cursor = 0;
  for (int64_t parent_row = 0; parent_row < parent_count; ++parent_row) {
    offsets[static_cast<size_t>(parent_row)] = child_cursor;
    while (child_cursor < child_count &&
           row_prefix_matches(parent_accessor, parent_row, child_accessor,
                              child_cursor, parent_columns)) {
      ++child_cursor;
    }
  }

  offsets[static_cast<size_t>(parent_count)] = child_cursor;
  return offsets;
}

std::pair<torch::Tensor, torch::Tensor>
pad_by_offsets(const torch::Tensor &flat_sequences,
               const std::vector<int64_t> &offsets, int64_t dimension_out) {
  const auto batch_size =
      static_cast<int64_t>(offsets.empty() ? 0 : offsets.size() - 1);
  const auto options =
      torch::TensorOptions().device(flat_sequences.device()).dtype(flat_sequences.dtype());
  auto mask_options =
      torch::TensorOptions().device(flat_sequences.device()).dtype(torch::kBool);

  if (batch_size == 0) {
    return {torch::empty({0, 0, dimension_out}, options),
            torch::empty({0, 0}, mask_options)};
  }

  int64_t max_sequence_length = 0;
  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    max_sequence_length = std::max(
        max_sequence_length, offsets[static_cast<size_t>(batch_index + 1)] -
                                 offsets[static_cast<size_t>(batch_index)]);
  }

  auto padded =
      torch::zeros({batch_size, max_sequence_length, dimension_out}, options);
  auto valid_token_mask =
      torch::zeros({batch_size, max_sequence_length}, mask_options);
  if (max_sequence_length == 0) {
    return {padded, valid_token_mask};
  }

  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    const auto start = offsets[static_cast<size_t>(batch_index)];
    const auto end = offsets[static_cast<size_t>(batch_index + 1)];
    const auto length = end - start;
    if (length == 0) {
      continue;
    }

    padded.index_put_(
        {batch_index, torch::indexing::Slice(0, length)},
        flat_sequences.index({torch::indexing::Slice(start, end)}));
    valid_token_mask.index_put_(
        {batch_index, torch::indexing::Slice(0, length)}, true);
  }

  return {padded, valid_token_mask};
}

torch::Tensor make_padding_attention_mask(const torch::Tensor &valid_token_mask,
                                          torch::Dtype dtype) {
  const auto sequence_length = valid_token_mask.size(1);
  auto attention_mask = torch::zeros(
      {valid_token_mask.size(0), 1, sequence_length, sequence_length},
      torch::TensorOptions().device(valid_token_mask.device()).dtype(dtype));
  auto invalid_key_mask =
      (~valid_token_mask)
          .unsqueeze(1)
          .unsqueeze(1)
          .expand({-1, 1, sequence_length, -1});
  return attention_mask.masked_fill(
      invalid_key_mask,
      static_cast<double>(std::numeric_limits<float>::lowest()));
}

torch::Tensor masked_sequence_sum(const torch::Tensor &sequence_tensor,
                                  const torch::Tensor &valid_token_mask) {
  return (sequence_tensor *
          valid_token_mask.unsqueeze(-1).to(sequence_tensor.dtype()))
      .sum(1);
}

} // namespace tensor_utils
