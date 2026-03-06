#ifndef NORMALIZED_LINEAR_H
#define NORMALIZED_LINEAR_H

#include "../include/SaveLoadMixin.h"
#include <torch/torch.h>

struct NormalizedLinearImpl : torch::nn::Module,
                              SaveLoadMixin<NormalizedLinearImpl> {
  NormalizedLinearImpl(int64_t d_in, int64_t d_out, double divisor = 400.0,
                       torch::Device device = torch::kCPU,
                       torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const torch::Tensor &x);

private:
  double divisor_;
  torch::nn::Linear linear_{nullptr};
};

TORCH_MODULE(NormalizedLinear);

#endif
