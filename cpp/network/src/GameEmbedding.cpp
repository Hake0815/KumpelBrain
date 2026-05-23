#include "network/include/GameEmbedding.h"

#include "network/include/SharedInstructionEmbeddingsFactory.h"

GameEmbeddingImpl::GameEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    shared_embedding_holder_ =
        register_module("shared_embedding_holder", SharedEmbeddingHolder(dimension_out, device, dtype));
    const auto shared_instruction_embeddings = create_shared_instruction_embeddings(
        *this, shared_embedding_holder_.ptr(), dimension_out, device, dtype);
    instruction_data_embedding_ = shared_instruction_embeddings.instruction_data_embedding;
    instruction_embedding_ = shared_instruction_embeddings.instruction_embedding;
    condition_embedding_ = shared_instruction_embeddings.condition_embedding;
    attack_embedding_ = shared_instruction_embeddings.attack_embedding;
    ability_embedding_ = shared_instruction_embeddings.ability_embedding;

    game_state_embedding_ = register_module(
        "game_state_embedding",
        GameStateEmbedding(shared_embedding_holder_.ptr(), dimension_out, shared_instruction_embeddings, device, dtype));
    game_interaction_embedding_ = register_module(
        "game_interaction_embedding",
        GameInteractionEmbedding(shared_embedding_holder_.ptr(), dimension_out, shared_instruction_embeddings, device,
                                 dtype));
    to(device, dtype);
}

std::pair<torch::Tensor, torch::Tensor> GameEmbeddingImpl::embedGameState(const ProtoBufGameState& game_state) {
    return game_state_embedding_(game_state);
}

torch::Tensor GameEmbeddingImpl::embedGameInteraction(const std::vector<ProtoBufGameInteraction>& game_interactions,
                                                    torch::Tensor card_indices, torch::Tensor cards) {
    return game_interaction_embedding_(game_interactions, card_indices, cards);
}
