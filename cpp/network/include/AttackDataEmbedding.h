#ifndef ATTACK_DATA_EMBEDDING_H
#define ATTACK_DATA_EMBEDDING_H

#include "../include/NormalizedLinear.h"
#include "../include/SaveLoadMixin.h"
#include <torch/torch.h>

struct AttackDataEmbeddingImpl : torch::nn::Module,
                                 SaveLoadMixin<AttackDataEmbeddingImpl> {
  AttackDataEmbeddingImpl(int64_t dimension_out,
                          torch::Device device = torch::kCPU,
                          torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(const torch::Tensor &attack_data);

private:
  torch::nn::Embedding attack_target_embedding_{nullptr};
  NormalizedLinear self_damage_embedding_{nullptr};
};

TORCH_MODULE(AttackDataEmbedding);

#endif
