#ifndef CARD_POSITION_EMBEDDING_H
#define CARD_POSITION_EMBEDDING_H

#include <torch/nn/modules/embedding.h>
#include <torch/torch.h>

#include <vector>

#include "network/include/NormalizedLinear.h"
#include "network/include/SaveLoadMixin.h"
#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufCardState = gamecore::serialization::ProtoBufCardState;

/// Learns `ProtoBufCardState.position` as a single [batch, dimension_out] tensor per row.
struct CardPositionEmbeddingImpl : torch::nn::Module, SaveLoadMixin<CardPositionEmbeddingImpl> {
    CardPositionEmbeddingImpl(int64_t dimension_out, torch::Device device = torch::kCPU,
                              torch::Dtype dtype = torch::kFloat);

    /// Shape [batch_size, dimension_out]. Rows are zero when `!has_position()`.
    torch::Tensor forward(const std::vector<ProtoBufCardState>& card_state_batch);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;
    torch::nn::Embedding owner_embedding_{nullptr};
    torch::nn::Embedding possible_position_embedding_{nullptr};
    torch::nn::Embedding opponent_position_knowledge_embedding_{nullptr};
    NormalizedLinear top_deck_position_index_embedding_{nullptr};
};

TORCH_MODULE(CardPositionEmbedding);

#endif
