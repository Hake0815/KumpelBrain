#ifndef CARD_STATE_EMBEDDING_H
#define CARD_STATE_EMBEDDING_H

#include <torch/torch.h>

#include <array>

#include "network/include/CardEmbedding.h"
#include "network/include/CardPositionEmbedding.h"
#include "network/include/NormalizedLinear.h"
#include "network/include/SaveLoadMixin.h"
#include "network/include/SharedEmbeddingHolder.h"

using ProtoBufCardState = gamecore::serialization::ProtoBufCardState;

static constexpr int64_t kNumRelationTypes = 7;
static constexpr int64_t kNumRgcnLayers = 2;
static constexpr int64_t kNumDegreeFeatures = 6;

struct RgcnLayerWeights {
    torch::nn::Linear self_loop{nullptr};
    torch::nn::Linear evolves_from{nullptr};
    torch::nn::Linear evolves_into{nullptr};
    torch::nn::Linear evolved_from{nullptr};
    torch::nn::Linear evolved_into{nullptr};
    torch::nn::Linear energy_attached_to{nullptr};
    torch::nn::Linear attached_energy_cards{nullptr};
};

/// Embeds a batch of `ProtoBufCardState` into shape [batch, dimension_out].
struct CardStateEmbeddingImpl : torch::nn::Module, SaveLoadMixin<CardStateEmbeddingImpl> {
    CardStateEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                           torch::Dtype dtype = torch::kFloat);

    torch::Tensor forward(const google::protobuf::RepeatedPtrField<ProtoBufCardState>& card_state_batch);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    SharedEmbeddingHolder shared_embedding_holder_{nullptr};
    CardEmbedding card_embedding_{nullptr};
    CardPositionEmbedding position_embedding_{nullptr};
    torch::nn::Linear card_position_gate_{nullptr};
    NormalizedLinear degree_count_embedding_{nullptr};
    std::array<RgcnLayerWeights, kNumRgcnLayers> rgcn_layers_;

    torch::Tensor aggregate_one_layer(const torch::Tensor& node_emb, const AdjacencyMatrices& adj,
                                      RgcnLayerWeights& weights);
    static torch::Tensor compute_degree_features(const AdjacencyMatrices& adj, int64_t num_cards, torch::Device device,
                                                 torch::Dtype dtype);
};

TORCH_MODULE(CardStateEmbedding);

#endif
