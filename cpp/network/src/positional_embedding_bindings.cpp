#include "../include/PositionalEmbedding.h"
#include <torch/extension.h>

// Expose class to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<PositionalEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<PositionalEmbeddingImpl>>(
      m, "PositionalEmbedding")
      .def(pybind11::init<int64_t, double, int64_t>())
      .def("forward", &PositionalEmbeddingImpl::forward);
}