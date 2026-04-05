#ifndef ATTACK_DATA_EMBEDDING_H
#define ATTACK_DATA_EMBEDDING_H

#include <torch/torch.h>

#include "../include/NormalizedLinear.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"

struct AttackDataEmbeddingImpl : torch::nn::Module, SaveLoadMixin<AttackDataEmbeddingImpl> {
    AttackDataEmbeddingImpl(std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out,
                            torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const torch::Tensor& attack_data);

   private:
    torch::nn::Embedding attack_target_embedding_{nullptr};
    NormalizedLinear damage_embedding_{nullptr};
};

TORCH_MODULE(AttackDataEmbedding);

#endif
