#include "../include/AttackDataEmbedding.h"

AttackDataEmbeddingImpl::AttackDataEmbeddingImpl(int64_t dimension_out,
                                                 torch::Device device,
                                                 torch::Dtype dtype) {
  attack_target_embedding_ = register_module(
      "attack_target_embedding", torch::nn::Embedding(1, dimension_out));
  self_damage_embedding_ =
      register_module("self_damage_embedding",
                      NormalizedLinear(1, dimension_out, 400.0, device, dtype));

  to(device, dtype);
}

torch::Tensor AttackDataEmbeddingImpl::forward(const torch::Tensor &attack_data) {
  auto attack_target = attack_data.index({torch::indexing::Slice(), 0});
  auto damage = attack_data.index({torch::indexing::Slice(), 1}).unsqueeze(1);
  return attack_target_embedding_(attack_target) + self_damage_embedding_(damage);
}
