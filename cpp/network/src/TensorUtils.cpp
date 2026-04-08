#include "../include/TensorUtils.h"

#include <ATen/Context.h>
#include <c10/util/ArrayRef.h>

#include <limits>
#include <stdexcept>

namespace tensor_utils {

namespace {

torch::Tensor build_offsets_from_counts(const torch::Tensor& counts) {
    auto zero = torch::zeros({1}, counts.options());
    return torch::cat({zero, counts.cumsum(0)});
}

torch::Tensor make_long_options_like(const torch::Tensor& tensor) {
    return tensor.to(torch::TensorOptions().device(tensor.device()).dtype(torch::kInt64));
}

torch::Tensor build_offsets_from_sorted_group_ids(const torch::Tensor& group_ids, int64_t num_groups) {
    auto options = torch::TensorOptions().device(group_ids.device()).dtype(torch::kInt64);
    if (num_groups <= 0) {
        return torch::zeros({1}, options);
    }

    auto flattened_group_ids = make_long_options_like(group_ids).contiguous().view({-1});
    auto group_starts = torch::searchsorted(flattened_group_ids, torch::arange(num_groups, options),
                                            /*out_int32=*/false, /*right=*/false);
    auto total_count = torch::full({1}, flattened_group_ids.size(0), options);
    return torch::cat({group_starts, total_count});
}

torch::Tensor build_offsets_from_group_ids(const torch::Tensor& group_ids, int64_t num_groups) {
    auto options = torch::TensorOptions().device(group_ids.device()).dtype(torch::kInt64);
    if (num_groups <= 0) {
        return torch::zeros({1}, options);
    }

    auto flattened_group_ids = make_long_options_like(group_ids).view({-1});

    // Prefer the cheaper histogram path unless CUDA determinism is explicitly
    // enabled, because bincount has no deterministic CUDA implementation.
    if (!group_ids.is_cuda() || !at::globalContext().deterministicAlgorithms()) {
        auto counts = torch::bincount(flattened_group_ids, torch::Tensor(), num_groups);
        return build_offsets_from_counts(counts);
    }

    return build_offsets_from_sorted_group_ids(flattened_group_ids, num_groups);
}

}  // namespace

torch::Tensor tensor_from_2d_int64(const std::vector<std::vector<int64_t>>& values, std::optional<torch::Device> device,
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

    for (const auto& row : values) {
        if (static_cast<int64_t>(row.size()) != width) {
            throw std::invalid_argument("Inconsistent row size in 2D tensor input");
        }
        flat.insert(flat.end(), row.begin(), row.end());
    }

    return torch::tensor(flat, options).view({static_cast<int64_t>(values.size()), width});
}

torch::Tensor build_contiguous_offsets(const torch::Tensor& group_indices, int64_t num_groups) {
    return build_offsets_from_group_ids(group_indices, num_groups);
}

torch::Tensor build_parent_offsets(const torch::Tensor& parent_row_ids, int64_t num_parents) {
    return build_offsets_from_group_ids(parent_row_ids, num_parents);
}

torch::Tensor local_positions_from_batch_offsets(const torch::Tensor& batch_offsets, int64_t num_rows) {
    if (num_rows <= 0) {
        return torch::empty({0}, torch::TensorOptions().device(batch_offsets.device()).dtype(torch::kInt64));
    }
    const auto batch_size = batch_offsets.size(0) - 1;
    auto row_ids = torch::arange(num_rows, batch_offsets.options());
    auto end_offsets = batch_offsets.slice(0, 1, batch_size + 1).contiguous();
    auto group_ids = torch::searchsorted(end_offsets, row_ids, /*out_int32=*/false, /*right=*/true);
    return row_ids - batch_offsets.index_select(0, group_ids);
}

std::pair<torch::Tensor, torch::Tensor> pad_by_offsets(const torch::Tensor& flat_sequences,
                                                       const torch::Tensor& offsets, int64_t dimension_out,
                                                       std::optional<int64_t> max_sequence_length_opt) {
    const auto batch_size = offsets.size(0) == 0 ? 0 : offsets.size(0) - 1;
    const auto options = torch::TensorOptions().device(flat_sequences.device()).dtype(flat_sequences.dtype());
    auto mask_options = torch::TensorOptions().device(flat_sequences.device()).dtype(torch::kBool);

    if (batch_size == 0) {
        return {torch::empty({0, 0, dimension_out}, options), torch::empty({0, 0}, mask_options)};
    }

    auto lengths = offsets.slice(0, 1, batch_size + 1) - offsets.slice(0, 0, batch_size);
    int64_t max_sequence_length;
    if (max_sequence_length_opt.has_value()) {
        max_sequence_length = *max_sequence_length_opt;
    } else {
        max_sequence_length = lengths.max().item<int64_t>();
    }

    auto padded = torch::zeros({batch_size, max_sequence_length, dimension_out}, options);
    auto valid_token_mask = torch::zeros({batch_size, max_sequence_length}, mask_options);
    if (max_sequence_length == 0) {
        return {padded, valid_token_mask};
    }

    auto row_ids = torch::arange(flat_sequences.size(0), offsets.options());
    auto end_offsets = offsets.slice(0, 1, batch_size + 1).contiguous();
    auto group_ids = torch::searchsorted(end_offsets, row_ids, /*out_int32=*/false,
                                         /*right=*/true);
    auto positions = row_ids - offsets.index_select(0, group_ids);

    padded.index_put_({group_ids, positions}, flat_sequences);
    valid_token_mask.index_put_({group_ids, positions}, true);

    return {padded, valid_token_mask};
}

torch::Tensor make_padding_attention_mask(const torch::Tensor& valid_token_mask, int64_t query_seq_len,
                                          torch::Dtype dtype) {
    const auto sequence_length = valid_token_mask.size(1);
    auto attention_mask = torch::zeros({valid_token_mask.size(0), 1, query_seq_len, sequence_length},
                                       torch::TensorOptions().device(valid_token_mask.device()).dtype(dtype));
    auto invalid_key_mask = (~valid_token_mask).unsqueeze(1).unsqueeze(1).expand({-1, 1, query_seq_len, -1});
    return attention_mask.masked_fill(invalid_key_mask, static_cast<double>(std::numeric_limits<float>::lowest()));
}

torch::Tensor masked_sequence_sum(const torch::Tensor& sequence_tensor, const torch::Tensor& valid_token_mask) {
    return (sequence_tensor * valid_token_mask.unsqueeze(-1).to(sequence_tensor.dtype())).sum(1);
}

}  // namespace tensor_utils
