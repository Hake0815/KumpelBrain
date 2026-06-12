#ifndef GAME_STATE_BATCH_H
#define GAME_STATE_BATCH_H

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "network/src/serialization/gamecore_serialization.pb.h"

using ProtoBufCardState = gamecore::serialization::ProtoBufCardState;
using ProtoBufGameState = gamecore::serialization::ProtoBufGameState;

/// Cumulative card counts per game: size B+1, first entry 0.
struct CardSegmentInfo {
    int64_t batch_size = 0;
    int64_t total_cards = 0;
    int64_t max_cards_per_game = 0;
    std::vector<int64_t> offsets;
};

CardSegmentInfo build_card_segments(const std::vector<ProtoBufGameState>& game_states);

void append_card_states(const std::vector<ProtoBufGameState>& game_states,
                        google::protobuf::RepeatedPtrField<ProtoBufCardState>& out_cards, CardSegmentInfo& segments);

/// Padded deck-id lookup per game: shape [B, max_deck_id_plus_one], -1 where absent.
torch::Tensor build_batched_card_indices(const std::vector<std::vector<int64_t>>& per_game_card_indices,
                                         torch::Device device);

/// Flat lookup for deck-id resolution across a batch: size B * max_deck_id_plus_one.
/// Entry at g * stride + deck_id is g * max_cards_per_game + local_row, or -1.
torch::Tensor globalize_card_indices_for_resolution(const torch::Tensor& card_indices_batched, int64_t max_cards_per_game);

/// Offset deck ids by game_index * deck_id_stride for batched resolution.
torch::Tensor offset_deck_ids_by_game(const torch::Tensor& deck_ids, const torch::Tensor& game_indices,
                                      int64_t deck_id_stride);

/// Expand one game index per deck-id entry using ragged row lengths and per-row interaction indices.
torch::Tensor expand_game_indices_for_ragged_rows(const torch::Tensor& row_lengths,
                                                  const torch::Tensor& row_interaction_indices,
                                                  const torch::Tensor& interaction_game_indices);

/// Flattened cards and globalized deck-id lookup for batched interaction embedding.
struct BatchedCardResolution {
    torch::Tensor flat_cards;
    torch::Tensor global_card_indices;
    int64_t deck_id_stride = 0;
};

BatchedCardResolution prepare_batched_card_resolution(const torch::Tensor& card_indices_batched,
                                                      const torch::Tensor& cards_batched);

#endif
