#ifndef ENERGY_TYPE_EMBEDDING_H
#define ENERGY_TYPE_EMBEDDING_H

#include <torch/nn/modules/embedding.h>
#include <torch/torch.h>

#include "../include/SaveLoadMixin.h"

enum EnergyTypeContext {
    ATTACK_COST,
    WEAKNESS,
    RESISTANCE,
    ENERGY_TYPE,
    POKEMON_TYPE,
    ENERGY_PROVIDED,
    ENERGY_ATTACHED
};
const int64_t NUMBER_OF_ENERGY_TYPE_CONTEXTS = 7;

struct EnergyTypeEmbeddingImpl : torch::nn::Module, SaveLoadMixin<EnergyTypeEmbeddingImpl> {
    EnergyTypeEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                            torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const torch::Tensor& energy_type_batch, const torch::Tensor& energy_type_contexts);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::nn::Embedding energy_type_embedding_{nullptr};
    torch::nn::Embedding context_embedding_{nullptr};
};

TORCH_MODULE(EnergyTypeEmbedding);

#endif
