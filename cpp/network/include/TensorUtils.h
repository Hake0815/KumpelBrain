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

} // namespace tensor_utils

#endif
