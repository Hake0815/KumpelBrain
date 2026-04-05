#include "../include/PositionalEmbedding.h"
#include <torch/torch.h>

using torch::indexing::None;
using torch::indexing::Slice;

PositionalEmbeddingImpl::PositionalEmbeddingImpl(int64_t d_model,
                                                 double dropout,
                                                 int64_t max_len,
                                                 torch::Device device,
                                                 torch::Dtype dtype) {
  // Dropout
  dropout_ = register_module("dropout", torch::nn::Dropout(dropout));

  // Options (device + dtype)
  auto options = torch::TensorOptions().device(device).dtype(dtype);

  // position = torch.arange(max_len).unsqueeze(1)
  auto position = torch::arange(max_len, options).unsqueeze(1);

  // div_term
  auto div_term = torch::exp(torch::arange(1, d_model + 1, options) *
                             (-std::log(10000.0) / d_model));

  // pe = torch.zeros(1, max_len, d_model)
  auto positional_embedding = torch::zeros({1, max_len, d_model}, options);

  // pe[0, :, 0::2] = sin(...)
  positional_embedding.index_put_(
      {0, Slice(), Slice(0, None, 2)},
      torch::sin(position * div_term.index({Slice(0, None, 2)})));

  // pe[0, :, 1::2] = cos(...)
  positional_embedding.index_put_(
      {0, Slice(), Slice(1, None, 2)},
      torch::cos(position * div_term.index({Slice(1, None, 2)})));

  // register_buffer("pe", pe)
  positional_embedding_ = register_buffer("pe", positional_embedding);
}

torch::Tensor PositionalEmbeddingImpl::forward(const torch::Tensor &x) {
  auto positional_embedding_slice =
      positional_embedding_.index({Slice(), Slice(0, x.size(1)), Slice()});

  return dropout_(x + positional_embedding_slice);
}

torch::Tensor
PositionalEmbeddingImpl::forward_packed(const torch::Tensor &x,
                                        const torch::Tensor &local_positions) {
  if (x.size(0) == 0) {
    return x;
  }
  TORCH_CHECK(x.dim() == 2, "forward_packed: x must be [N, d_model]");
  TORCH_CHECK(local_positions.dim() == 1 &&
                  local_positions.size(0) == x.size(0),
              "forward_packed: local_positions must be [N] matching x.size(0)");
  auto pos = local_positions.to(torch::kLong);
  auto pe_rows = positional_embedding_.index({0, pos, Slice()});
  return dropout_(x + pe_rows);
}