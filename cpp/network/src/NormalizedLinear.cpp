#include "../include/NormalizedLinear.h"

NormalizedLinearImpl::NormalizedLinearImpl(int64_t d_in, int64_t d_out,
                                           double divisor, torch::Device device,
                                           torch::Dtype dtype)
    : divisor_(divisor) {
  linear_ = register_module(
      "linear",
      torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(false)));
  to(device, dtype);
}

torch::Tensor NormalizedLinearImpl::forward(const torch::Tensor &x) {
  return linear_(x / divisor_);
}
