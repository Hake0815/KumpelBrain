#ifndef CARD_STATE_EMBEDDING_H
#define CARD_STATE_EMBEDDING_H

#include <torch/nn/modules/linear.h>

#include "network/include/CardEmbedding.h"
using ProtoBufCardState = gamecore::serialization::ProtoBufCardState;

/// Embeds a batch of `ProtoBufCardState` into shape [batch, dimension_out].
struct CardStateEmbeddingImpl : torch::nn::Module, SaveLoadMixin<CardStateEmbeddingImpl> {
    CardStateEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                           torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const std::vector<ProtoBufCardState>& card_state_batch);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    CardEmbedding card_embedding_{nullptr};
    torch::nn::Linear evolves_from_weights_{nullptr};
    torch::nn::Linear evolves_into_weights_{nullptr};
    torch::nn::Linear evolved_from_weights_{nullptr};
    torch::nn::Linear evolved_into_weights_{nullptr};
    torch::nn::Linear energy_attached_to_weights_{nullptr};
    torch::nn::Linear attached_energy_cards_weights_{nullptr};
};

TORCH_MODULE(CardStateEmbedding);

#endif
