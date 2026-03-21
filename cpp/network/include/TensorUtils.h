#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <optional>
#include <vector>

#include <torch/torch.h>

namespace tensor_utils {

torch::Tensor
tensor_from_2d_int64(const std::vector<std::vector<int64_t>> &values,
                     std::optional<torch::Device> device = std::nullopt,
                     std::optional<torch::Dtype> dtype = std::nullopt);

torch::Tensor build_contiguous_offsets(const torch::Tensor &group_indices,
                                       int64_t num_groups);

torch::Tensor build_parent_offsets(const torch::Tensor &parent_row_ids,
                                   int64_t num_parents);

std::pair<torch::Tensor, torch::Tensor>
pad_by_offsets(const torch::Tensor &flat_sequences,
               const torch::Tensor &offsets, int64_t dimension_out);

torch::Tensor make_padding_attention_mask(const torch::Tensor &valid_token_mask,
                                          torch::Dtype dtype);

torch::Tensor masked_sequence_sum(const torch::Tensor &sequence_tensor,
                                  const torch::Tensor &valid_token_mask);

} // namespace tensor_utils

#endif
