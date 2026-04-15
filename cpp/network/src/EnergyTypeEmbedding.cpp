#include "../include/EnergyTypeEmbedding.h"

#include <ATen/ops/full_like.h>

EnergyTypeEmbeddingImpl::EnergyTypeEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    energy_type_embedding_ = register_module("energy_type_embedding", torch::nn::Embedding(11, dimension_out));
    context_embedding_ =
        register_module("context_embedding", torch::nn::Embedding(NUMBER_OF_ENERGY_TYPE_CONTEXTS, dimension_out));
    to(device, dtype);
}

torch::Tensor EnergyTypeEmbeddingImpl::forward(const std::vector<int64_t>& energy_type_batch,
                                               const EnergyTypeContext& energy_type_context) {
    const auto index_options = torch::TensorOptions().device(device_).dtype(torch::kLong);

    auto energy_type_batch_tensor = torch::tensor(energy_type_batch, index_options);
    auto energy_type_context_tensor =
        torch::full_like(energy_type_batch_tensor, static_cast<int64_t>(energy_type_context), index_options);

    return energy_type_embedding_(energy_type_batch_tensor) + context_embedding_(energy_type_context_tensor);
}