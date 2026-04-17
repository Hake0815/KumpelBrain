#ifndef SHARED_EMBEDDING_HOLDER_H
#define SHARED_EMBEDDING_HOLDER_H

#include <torch/torch.h>

#include "../include/EnergyTypeEmbedding.h"
#include "../include/NormalizedLinear.h"
#include "../include/PositionalEmbedding.h"
#include "../include/SaveLoadMixin.h"

struct SharedEmbeddingHolderImpl : torch::nn::Module, SaveLoadMixin<SharedEmbeddingHolderImpl> {
    SharedEmbeddingHolderImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                              torch::Dtype dtype = torch::kFloat);

    torch::nn::Embedding card_type_embedding_{nullptr};
    torch::nn::Embedding card_subtype_embedding_{nullptr};
    NormalizedLinear hp_embedding_{nullptr};
    NormalizedLinear card_amount_range_embedding_{nullptr};
    torch::nn::Embedding card_position_embedding_{nullptr};
    torch::nn::Embedding player_target_embedding_{nullptr};
    PositionalEmbedding position_embedding_{nullptr};
    NormalizedLinear damage_embedding_{nullptr};
    EnergyTypeEmbedding energy_type_embedding_{nullptr};
};

TORCH_MODULE(SharedEmbeddingHolder);

#endif
