#include "network/include/EnergyTypeEmbedding.h"

#include <cstdint>

#include "network/include/SharedConstants.h"

EnergyTypeEmbeddingImpl::EnergyTypeEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    energy_type_embedding_ =
        register_module("energy_type_embedding", torch::nn::Embedding(NUMBER_ENERGY_TYPES, dimension_out));
    context_embedding_ =
        register_module("context_embedding", torch::nn::Embedding(NUMBER_OF_ENERGY_TYPE_CONTEXTS, dimension_out));
    to(device, dtype);
}

torch::Tensor EnergyTypeEmbeddingImpl::forward(const torch::Tensor& energy_type_batch,
                                               const torch::Tensor& energy_type_contexts) {
    return energy_type_embedding_(energy_type_batch) + context_embedding_(energy_type_contexts);
}