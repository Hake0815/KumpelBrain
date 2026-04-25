#include "network/include/PlayerStateEmbedding.h"

#include <algorithm>

#include "network/include/AttentionUtils.h"
#include "network/include/SharedConstants.h"
#include "network/include/TensorUtils.h"

namespace {

constexpr int64_t kNumPlayers = 2;
constexpr int64_t kNumBooleanFeatures = 3;
constexpr int64_t kNumCountFeatures = 6;
constexpr int64_t kNumBaseTokens = kNumBooleanFeatures + kNumCountFeatures;

}  // namespace

PlayerStateEmbeddingImpl::PlayerStateEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    is_active_embedding_ = register_module("is_active_embedding", torch::nn::Embedding(2, dimension_out));
    is_attacking_embedding_ = register_module("is_attacking_embedding", torch::nn::Embedding(2, dimension_out));
    knows_his_prizes_embedding_ = register_module("knows_his_prizes_embedding", torch::nn::Embedding(2, dimension_out));
    hand_count_embedding_ =
        register_module("hand_count_embedding", NormalizedLinear(1, dimension_out, LARGE_HAND_COUNT, device, dtype));
    deck_count_embedding_ =
        register_module("deck_count_embedding", NormalizedLinear(1, dimension_out, DECK_SIZE, device, dtype));
    prizes_count_embedding_ =
        register_module("prizes_count_embedding", NormalizedLinear(1, dimension_out, MAX_PRIZES, device, dtype));
    bench_count_embedding_ =
        register_module("bench_count_embedding", NormalizedLinear(1, dimension_out, BENCH_SIZE, device, dtype));
    discard_pile_count_embedding_ = register_module(
        "discard_pile_count_embedding", NormalizedLinear(1, dimension_out, LARGE_DISCARD_PILE_SIZE, device, dtype));
    turn_counter_embedding_ =
        register_module("turn_counter_embedding", NormalizedLinear(1, dimension_out, LARGE_TURNCOUNTER, device, dtype));
    player_turn_traits_embedding_ =
        register_module("player_turn_traits_embedding", torch::nn::Embedding(NUMBER_PLAYER_TURN_TRAITS, dimension_out));
    queries_embedding_ = register_module("queries_embedding", torch::nn::Embedding(2, dimension_out));
    multi_head_attention_ =
        register_module("multi_head_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0, true, device_, dtype_));
    index_options_ = torch::TensorOptions().device(device_).dtype(torch::kLong);
    float_options_ = torch::TensorOptions().device(device_).dtype(dtype_);
    query_indices_ = torch::tensor({static_cast<int64_t>(0), static_cast<int64_t>(1)}, index_options_);
    base_token_mask_ =
        torch::ones({kNumPlayers, kNumBaseTokens}, torch::TensorOptions().device(device_).dtype(torch::kBool));
    to(device, dtype);
}

torch::Tensor PlayerStateEmbeddingImpl::forward(const ProtoBufPlayerState& self_player_state,
                                                const ProtoBufPlayerState& opponent_player_state) {
    auto features = collect_features(self_player_state, opponent_player_state);
    auto staged = stage_features(features);
    auto base_tokens = embed_base_tokens(staged);
    auto trait_tokens = embed_trait_tokens(staged, features.max_player_turn_traits);
    auto padded = torch::cat({base_tokens, trait_tokens.tokens}, 1);
    auto valid_token_mask = torch::cat({base_token_mask_, trait_tokens.mask}, 1);
    auto queries = queries_embedding_(query_indices_).unsqueeze(1);
    return attention_utils::masked_attention_pooling(multi_head_attention_, queries, padded, valid_token_mask);
}

PlayerStateEmbeddingImpl::PlayerStateFeatures PlayerStateEmbeddingImpl::collect_features(
    const ProtoBufPlayerState& self_player_state, const ProtoBufPlayerState& opponent_player_state) const {
    PlayerStateFeatures features;
    features.boolean_indices.reserve(kNumPlayers * kNumBooleanFeatures);
    features.counts.reserve(kNumPlayers * kNumCountFeatures);
    features.player_turn_trait_offsets.reserve(kNumPlayers + 1);
    features.player_turn_trait_offsets.push_back(0);

    const auto append_player = [&](const ProtoBufPlayerState& player_state) {
        features.boolean_indices.push_back(static_cast<int64_t>(player_state.is_active()));
        features.boolean_indices.push_back(static_cast<int64_t>(player_state.is_attacking()));
        features.boolean_indices.push_back(static_cast<int64_t>(player_state.knows_his_prizes()));

        features.counts.push_back(static_cast<float>(player_state.hand_count()));
        features.counts.push_back(static_cast<float>(player_state.deck_count()));
        features.counts.push_back(static_cast<float>(player_state.prizes_count()));
        features.counts.push_back(static_cast<float>(player_state.bench_count()));
        features.counts.push_back(static_cast<float>(player_state.discard_pile_count()));
        features.counts.push_back(static_cast<float>(player_state.turn_counter()));

        features.max_player_turn_traits =
            std::max<int64_t>(features.max_player_turn_traits, player_state.player_turn_traits_size());
        for (const auto& turn_trait : player_state.player_turn_traits()) {
            features.player_turn_traits.push_back(static_cast<int64_t>(turn_trait));
        }
        features.player_turn_trait_offsets.push_back(static_cast<int64_t>(features.player_turn_traits.size()));
    };

    append_player(self_player_state);
    append_player(opponent_player_state);
    return features;
}

PlayerStateEmbeddingImpl::PlayerStateStagedTensors PlayerStateEmbeddingImpl::stage_features(
    const PlayerStateFeatures& features) const {
    PlayerStateStagedTensors staged;
    staged.boolean_indices =
        torch::tensor(features.boolean_indices, index_options_).view({kNumPlayers, kNumBooleanFeatures});
    staged.counts = torch::tensor(features.counts, float_options_).view({kNumPlayers, kNumCountFeatures});
    staged.player_turn_traits = torch::tensor(features.player_turn_traits, index_options_);
    staged.player_turn_trait_offsets = torch::tensor(features.player_turn_trait_offsets, index_options_);
    return staged;
}

torch::Tensor PlayerStateEmbeddingImpl::embed_base_tokens(const PlayerStateStagedTensors& staged) {
    auto is_active = is_active_embedding_(staged.boolean_indices.select(1, 0));
    auto is_attacking = is_attacking_embedding_(staged.boolean_indices.select(1, 1));
    auto knows_his_prizes = knows_his_prizes_embedding_(staged.boolean_indices.select(1, 2));

    auto hand_count = hand_count_embedding_(staged.counts.slice(1, 0, 1));
    auto deck_count = deck_count_embedding_(staged.counts.slice(1, 1, 2));
    auto prizes_count = prizes_count_embedding_(staged.counts.slice(1, 2, 3));
    auto bench_count = bench_count_embedding_(staged.counts.slice(1, 3, 4));
    auto discard_pile_count = discard_pile_count_embedding_(staged.counts.slice(1, 4, 5));
    auto turn_counter = turn_counter_embedding_(staged.counts.slice(1, 5, 6));

    return torch::stack({is_active, is_attacking, knows_his_prizes, hand_count, deck_count, prizes_count, bench_count,
                         discard_pile_count, turn_counter},
                        1);
}

PlayerStateEmbeddingImpl::PlayerTraitTokens PlayerStateEmbeddingImpl::embed_trait_tokens(
    const PlayerStateStagedTensors& staged, int64_t max_player_turn_traits) {
    auto embedded_traits = player_turn_traits_embedding_(staged.player_turn_traits);
    auto [padded_traits, trait_mask] =
        tensor_utils::pad_by_offsets(embedded_traits, staged.player_turn_trait_offsets, dimension_out_,
                                     max_player_turn_traits);
    return {padded_traits, trait_mask};
}
