#include "../include/AttackDataEmbedding.h"

AttackDataEmbeddingImpl::AttackDataEmbeddingImpl(std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
                                                 int64_t dimension_out, torch::Device device, torch::Dtype dtype) {
    attack_target_embedding_ = register_module("attack_target_embedding", torch::nn::Embedding(1, dimension_out));
    damage_embedding_ = shared_embedding_holder->damage_embedding_;

    to(device, dtype);
}

torch::Tensor AttackDataEmbeddingImpl::forward(const torch::Tensor& attack_data) {
    auto attack_target = attack_data.index({torch::indexing::Slice(), 0});
    auto damage = attack_data.index({torch::indexing::Slice(), 1}).unsqueeze(1);
    return attack_target_embedding_(attack_target) + damage_embedding_(damage);
}
