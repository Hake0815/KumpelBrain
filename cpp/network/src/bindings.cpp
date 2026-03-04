#include "../include/MultiHeadAttention.h"
#include "../include/PositionalEmbedding.h"
#include <torch/extension.h>

// Expose class to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<PositionalEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<PositionalEmbeddingImpl>>(
      m, "PositionalEmbedding")
      .def(pybind11::init<int64_t, double, int64_t>())
      .def("forward", &PositionalEmbeddingImpl::forward);
  pybind11::class_<MultiHeadAttentionImpl, torch::nn::Module,
                   std::shared_ptr<MultiHeadAttentionImpl>>(
      m, "MultiHeadAttention")
      .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, int64_t>())
      .def("forward", &MultiHeadAttentionImpl::forward)
      .def("save_weights", &MultiHeadAttentionImpl::save_weights)
      .def("load_weights", &MultiHeadAttentionImpl::load_weights);
}