#include "../include/AttackDataEmbedding.h"
#include "../include/MultiHeadAttention.h"
#include "../include/NormalizedLinear.h"
#include "../include/PositionalEmbedding.h"
#include "../include/SharedEmbeddingHolder.h"
#include <torch/extension.h>

// Expose class to Python
PYBIND11_MODULE(kumpel_embedding, m) {
  pybind11::module_::import("torch");
  pybind11::class_<PositionalEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<PositionalEmbeddingImpl>>(
      m, "PositionalEmbedding")
      .def(pybind11::init<int64_t, double, int64_t, torch::Device,
                          torch::Dtype>(),
           pybind11::arg("d_model"), pybind11::arg("dropout") = 0.1,
           pybind11::arg("max_len") = 5000,
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &PositionalEmbeddingImpl::forward);
  pybind11::class_<MultiHeadAttentionImpl, torch::nn::Module,
                   std::shared_ptr<MultiHeadAttentionImpl>>(
      m, "MultiHeadAttention")
      .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, int64_t, double,
                          bool, torch::Device, torch::Dtype>(),
           pybind11::arg("d_q"), pybind11::arg("d_k"), pybind11::arg("d_v"),
           pybind11::arg("d_head"), pybind11::arg("nheads"),
           pybind11::arg("dropout") = 0.0, pybind11::arg("bias") = true,
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &MultiHeadAttentionImpl::forward)
      .def("save_weights", &MultiHeadAttentionImpl::save_weights)
      .def("load_weights", &MultiHeadAttentionImpl::load_weights);
  pybind11::class_<NormalizedLinearImpl, torch::nn::Module,
                   std::shared_ptr<NormalizedLinearImpl>>(m, "NormalizedLinear")
      .def(pybind11::init<int64_t, int64_t, double, torch::Device,
                          torch::Dtype>(),
           pybind11::arg("d_in"), pybind11::arg("d_out"),
           pybind11::arg("divisor") = 400.0,
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &NormalizedLinearImpl::forward)
      .def("save_weights", &NormalizedLinearImpl::save_weights)
      .def("load_weights", &NormalizedLinearImpl::load_weights);
  pybind11::class_<SharedEmbeddingHolderImpl, torch::nn::Module,
                   std::shared_ptr<SharedEmbeddingHolderImpl>>(
      m, "SharedEmbeddingHolder")
      .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("save_weights", &SharedEmbeddingHolderImpl::save_weights)
      .def("load_weights", &SharedEmbeddingHolderImpl::load_weights);
  pybind11::class_<AttackDataEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<AttackDataEmbeddingImpl>>(
      m, "AttackDataEmbedding")
      .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &AttackDataEmbeddingImpl::forward)
      .def("save_weights", &AttackDataEmbeddingImpl::save_weights)
      .def("load_weights", &AttackDataEmbeddingImpl::load_weights);
}