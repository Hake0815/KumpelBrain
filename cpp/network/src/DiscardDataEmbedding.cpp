#include "../include/DiscardDataEmbedding.h"

#include "network/include/SharedConstants.h"

DiscardDataEmbeddingImpl::DiscardDataEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype) {
    target_source_embedding_ =
        register_module("target_source_embedding", torch::nn::Embedding(NUMBER_TARGET_SOURCES, dimension_out));
    to(device, dtype);
}

torch::Tensor DiscardDataEmbeddingImpl::forward(const torch::Tensor& discard_data) {
    return target_source_embedding_(discard_data);
}
