#include "../include/TensorUtils.h"

#include <stdexcept>

namespace tensor_utils {

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

} // namespace tensor_utils
