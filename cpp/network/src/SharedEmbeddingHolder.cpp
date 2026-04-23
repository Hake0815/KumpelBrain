#include "../include/SharedEmbeddingHolder.h"

#include "network/include/SharedConstants.h"

SharedEmbeddingHolderImpl::SharedEmbeddingHolderImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype) {
    card_type_embedding_ =
        register_module("card_type_embedding", torch::nn::Embedding(NUMBER_CARD_TYPES, dimension_out));
    card_subtype_embedding_ =
        register_module("card_subtype_embedding", torch::nn::Embedding(NUMBER_CARD_SUBTYPES, dimension_out));
    hp_embedding_ = register_module("hp_embedding", NormalizedLinear(1, dimension_out, 400.0, device, dtype));
    card_amount_range_embedding_ =
        register_module("card_amount_range_embedding", NormalizedLinear(2, dimension_out, 400.0, device, dtype));
    card_position_embedding_ = register_module(
        "card_position_embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(NUMBER_CARD_POSITIONS + 1, dimension_out).padding_idx(0)));
    player_target_embedding_ =
        register_module("player_target_embedding", torch::nn::Embedding(NUMBER_PLAYER_TARGETS, dimension_out));
    position_embedding_ =
        register_module("position_embedding", PositionalEmbedding(dimension_out, 0.1, 5000, device, dtype));
    damage_embedding_ = register_module("damage_embedding", NormalizedLinear(1, dimension_out, 400.0, device, dtype));
    energy_type_embedding_ =
        register_module("energy_type_embedding", EnergyTypeEmbedding(dimension_out, device, dtype));
    to(device, dtype);
}
