#include "network/include/GameEmbedding.h"

GameEmbeddingImpl::GameEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    shared_embedding_holder_ =
        register_module("shared_embedding_holder", SharedEmbeddingHolder(dimension_out, device, dtype));
    instruction_data_embedding_ = register_module(
        "instruction_data_embedding",
        InstructionDataEmbedding(shared_embedding_holder_.ptr(), dimension_out, device, dtype));
    instruction_embedding_ = register_module(
        "instruction_embedding",
        InstructionEmbedding(instruction_data_embedding_.ptr(), shared_embedding_holder_.ptr(), dimension_out, device,
                             dtype));
    condition_embedding_ = register_module(
        "condition_embedding",
        ConditionEmbedding(instruction_data_embedding_.ptr(), shared_embedding_holder_.ptr(), dimension_out, device,
                           dtype));
    attack_embedding_ = register_module("attack_embedding", AttackEmbedding(dimension_out, device, dtype));
    ability_embedding_ = register_module("ability_embedding", AbilityEmbedding(dimension_out, device, dtype));

    const SharedInstructionEmbeddings shared_instruction_embeddings{
        instruction_data_embedding_,
        instruction_embedding_,
        condition_embedding_,
        attack_embedding_,
        ability_embedding_,
    };

    game_state_embedding_ = register_module(
        "game_state_embedding",
        GameStateEmbedding(shared_embedding_holder_.ptr(), dimension_out, shared_instruction_embeddings, device, dtype));
    game_interaction_embedding_ = register_module(
        "game_interaction_embedding",
        GameInteractionEmbedding(shared_embedding_holder_.ptr(), dimension_out, shared_instruction_embeddings, device,
                                 dtype));
    to(device, dtype);
}

torch::Tensor GameEmbeddingImpl::embedGameState(const ProtoBufGameState& game_state) {
    return game_state_embedding_(game_state);
}

torch::Tensor GameEmbeddingImpl::embedGameInteraction(const std::vector<ProtoBufGameInteraction>& game_interactions,
                                                    torch::Tensor card_indices, torch::Tensor cards) {
    return game_interaction_embedding_(game_interactions, card_indices, cards);
}
