#include "network/include/GameStateBatch.h"

#include <ATen/ops/arange.h>
#include <ATen/ops/expand.h>
#include <ATen/ops/full.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/view.h>
#include <ATen/ops/where.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <algorithm>

CardSegmentInfo build_card_segments(const std::vector<ProtoBufGameState>& game_states) {
    CardSegmentInfo info;
    info.batch_size = static_cast<int64_t>(game_states.size());
    info.offsets.reserve(static_cast<size_t>(info.batch_size + 1));
    info.offsets.push_back(0);
    for (const auto& game_state : game_states) {
        const int64_t num_cards = static_cast<int64_t>(game_state.card_states_size());
        info.max_cards_per_game = std::max(info.max_cards_per_game, num_cards);
        info.total_cards += num_cards;
        info.offsets.push_back(info.total_cards);
    }
    return info;
}

void append_card_states(const std::vector<ProtoBufGameState>& game_states,
                        google::protobuf::RepeatedPtrField<ProtoBufCardState>& out_cards, CardSegmentInfo& segments) {
    segments = build_card_segments(game_states);
    out_cards.Clear();
    for (const auto& game_state : game_states) {
        for (const auto& card_state : game_state.card_states()) {
            *out_cards.Add() = card_state;
        }
    }
}

torch::Tensor build_batched_card_indices(const std::vector<std::vector<int64_t>>& per_game_card_indices,
                                         torch::Device device) {
    const int64_t batch_size = static_cast<int64_t>(per_game_card_indices.size());
    if (batch_size == 0) {
        return torch::empty({0, 0}, torch::TensorOptions().device(device).dtype(torch::kInt64));
    }
    int64_t max_lookup = 0;
    for (const auto& row : per_game_card_indices) {
        max_lookup = std::max(max_lookup, static_cast<int64_t>(row.size()));
    }
    auto options = torch::TensorOptions().device(device).dtype(torch::kInt64);
    auto out = torch::full({batch_size, max_lookup}, -1, options);
    for (int64_t g = 0; g < batch_size; ++g) {
        const auto& row = per_game_card_indices[static_cast<size_t>(g)];
        if (row.empty()) {
            continue;
        }
        out.slice(0, g, g + 1).slice(1, 0, static_cast<int64_t>(row.size())) =
            torch::tensor(row, options).unsqueeze(0);
    }
    return out;
}

torch::Tensor globalize_card_indices_for_resolution(const torch::Tensor& card_indices_batched,
                                                    int64_t max_cards_per_game) {
    if (card_indices_batched.numel() == 0) {
        return torch::empty({0}, card_indices_batched.options());
    }
    const int64_t batch_size = card_indices_batched.size(0);
    const int64_t lookup_width = card_indices_batched.size(1);
    const auto options = card_indices_batched.options();
    const auto flat_local = card_indices_batched.reshape({-1});
    const auto game_ids =
        torch::arange(batch_size, options).view({batch_size, 1}).expand({batch_size, lookup_width}).reshape({-1});
    const auto global_vals = game_ids * max_cards_per_game + flat_local;
    return torch::where(flat_local >= 0, global_vals, torch::full_like(flat_local, -1));
}

torch::Tensor offset_deck_ids_by_game(const torch::Tensor& deck_ids, const torch::Tensor& game_indices,
                                      int64_t deck_id_stride) {
    if (deck_ids.numel() == 0) {
        return deck_ids;
    }
    return deck_ids + game_indices * deck_id_stride;
}

torch::Tensor expand_game_indices_for_ragged_rows(const torch::Tensor& row_lengths,
                                                  const torch::Tensor& row_interaction_indices,
                                                  const torch::Tensor& interaction_game_indices) {
    if (row_lengths.numel() == 0) {
        return torch::empty({0}, row_lengths.options());
    }
    const auto games_per_row = interaction_game_indices.index_select(0, row_interaction_indices);
    return games_per_row.repeat_interleave(row_lengths);
}

BatchedCardResolution prepare_batched_card_resolution(const torch::Tensor& card_indices_batched,
                                                      const torch::Tensor& cards_batched) {
    BatchedCardResolution resolution;
    if (card_indices_batched.numel() == 0) {
        resolution.flat_cards = cards_batched.reshape({0, cards_batched.size(-1)});
        resolution.global_card_indices =
            torch::empty({0}, torch::TensorOptions().device(card_indices_batched.device()).dtype(torch::kInt64));
        return resolution;
    }
    TORCH_CHECK(cards_batched.numel() != 0,
                "prepare_batched_card_resolution: non-empty card_indices requires non-empty cards");
    const int64_t batch_size = card_indices_batched.size(0);
    const int64_t max_cards = cards_batched.size(1);
    resolution.deck_id_stride = card_indices_batched.size(1);
    resolution.flat_cards = cards_batched.reshape({batch_size * max_cards, cards_batched.size(2)});
    resolution.global_card_indices =
        globalize_card_indices_for_resolution(card_indices_batched, max_cards).to(card_indices_batched.device());
    return resolution;
}
