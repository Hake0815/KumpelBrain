#include "network/include/PlayerStateEmbedding.h"

#include <c10/core/ScalarType.h>

#include "network/include/AttentionUtils.h"
#include "network/include/SharedConstants.h"
#include "network/include/TensorUtils.h"

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
    to(device, dtype);
}

torch::Tensor PlayerStateEmbeddingImpl::forward(const ProtoBufPlayerState& self_player_state,
                                                const ProtoBufPlayerState& opponent_player_state) {
    auto embedded_self_player_state = embed_player_state(self_player_state);
    auto embedded_opponent_player_state = embed_player_state(opponent_player_state);

    const int64_t self_len = embedded_self_player_state.size(0);
    const int64_t opp_len = embedded_opponent_player_state.size(0);
    auto flat = torch::cat({embedded_self_player_state, embedded_opponent_player_state}, 0);
    auto offsets = torch::tensor({static_cast<int64_t>(0), self_len, self_len + opp_len}, index_options_);
    auto [padded, valid_token_mask] = tensor_utils::pad_by_offsets(flat, offsets, dimension_out_);

    auto queries = queries_embedding_(torch::tensor({0, 1}, index_options_)).unsqueeze(1);
    return attention_utils::masked_attention_pooling(multi_head_attention_, queries, padded, valid_token_mask);
}

torch::Tensor PlayerStateEmbeddingImpl::embed_player_state(const ProtoBufPlayerState& player_state) {
    auto is_active = is_active_embedding_(torch::tensor(player_state.is_active(), index_options_));
    auto is_attacking = is_attacking_embedding_(torch::tensor(player_state.is_attacking(), index_options_));
    auto knows_his_prizes = knows_his_prizes_embedding_(torch::tensor(player_state.knows_his_prizes(), index_options_));
    auto hand_count = hand_count_embedding_(torch::tensor(player_state.hand_count(), float_options_).unsqueeze(0));
    auto deck_count = deck_count_embedding_(torch::tensor(player_state.deck_count(), float_options_).unsqueeze(0));
    auto prizes_count =
        prizes_count_embedding_(torch::tensor(player_state.prizes_count(), float_options_).unsqueeze(0));
    auto bench_count = bench_count_embedding_(torch::tensor(player_state.bench_count(), float_options_).unsqueeze(0));
    auto discard_pile_count =
        discard_pile_count_embedding_(torch::tensor(player_state.discard_pile_count(), float_options_).unsqueeze(0));
    auto turn_counter =
        turn_counter_embedding_(torch::tensor(player_state.turn_counter(), float_options_).unsqueeze(0));

    auto tokens = torch::stack({is_active, is_attacking, knows_his_prizes, hand_count, deck_count, prizes_count,
                                bench_count, discard_pile_count, turn_counter});

    if (player_state.player_turn_traits_size() > 0) {
        std::vector<int64_t> turn_traits;
        for (auto& turn_trait : player_state.player_turn_traits()) {
            turn_traits.push_back(turn_trait);
        }
        auto player_turn_traits = player_turn_traits_embedding_(torch::tensor(turn_traits, index_options_));

        tokens = torch::cat({tokens, player_turn_traits}, 0);
    }
    return tokens;  // shape: (9 + number of turn traits, dimension_out)
}
