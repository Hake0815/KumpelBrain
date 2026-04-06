#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <torch/torch.h>

#include <optional>
#include <vector>

namespace tensor_utils {

torch::Tensor tensor_from_2d_int64(const std::vector<std::vector<int64_t>>& values,
                                   std::optional<torch::Device> device = std::nullopt,
                                   std::optional<torch::Dtype> dtype = std::nullopt);

torch::Tensor build_contiguous_offsets(const torch::Tensor& group_indices, int64_t num_groups);

/// For each flat row index 0..num_rows-1, position within its batch group (0,1,2,...)
/// batch_offsets has shape (num_groups + 1,) with cumulative counts per group.
torch::Tensor local_positions_from_batch_offsets(const torch::Tensor& batch_offsets, int64_t num_rows);

torch::Tensor build_parent_offsets(const torch::Tensor& parent_row_ids, int64_t num_parents);

/// When max_sequence_length is set (CPU precomputed), skips device sync from lengths.max().item().
std::pair<torch::Tensor, torch::Tensor> pad_by_offsets(const torch::Tensor& flat_sequences,
                                                       const torch::Tensor& offsets, int64_t dimension_out,
                                                       std::optional<int64_t> max_sequence_length = std::nullopt);

torch::Tensor make_padding_attention_mask(const torch::Tensor& valid_token_mask, int64_t query_seq_len,
                                          torch::Dtype dtype);

torch::Tensor masked_sequence_sum(const torch::Tensor& sequence_tensor, const torch::Tensor& valid_token_mask);

}  // namespace tensor_utils

#endif
