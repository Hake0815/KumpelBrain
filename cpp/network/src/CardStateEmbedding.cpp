#include "../include/CardStateEmbedding.h"

CardStateEmbeddingImpl::CardStateEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    card_embedding_ = register_module("card_embedding", CardEmbedding(dimension_out, device, dtype));
    evolves_from_weights_ = register_module("evolves_from_weights", torch::nn::Linear(dimension_out, dimension_out));
    evolves_into_weights_ = register_module("evolves_into_weights", torch::nn::Linear(dimension_out, dimension_out));
    evolved_from_weights_ = register_module("evolved_from_weights", torch::nn::Linear(dimension_out, dimension_out));
    evolved_into_weights_ = register_module("evolved_into_weights", torch::nn::Linear(dimension_out, dimension_out));
    energy_attached_to_weights_ =
        register_module("energy_attached_to_weights", torch::nn::Linear(dimension_out, dimension_out));
    attached_energy_cards_weights_ =
        register_module("attached_energy_cards_weights", torch::nn::Linear(dimension_out, dimension_out));
    to(device, dtype);
}

torch::Tensor CardStateEmbeddingImpl::forward(const std::vector<ProtoBufCardState>& card_state_batch) {
    std::vector<ProtoBufCard> cards;
    cards.reserve(card_state_batch.size());
    for (const auto& card_state : card_state_batch) {
        cards.push_back(card_state.card());
    }
    auto [card_embeddings, adjacency_matrices] = card_embedding_->forward(cards);

    return card_embeddings +
           adjacency_matrices.evolves_from_adjacency.matmul(evolves_from_weights_->forward(card_embeddings)) +
           adjacency_matrices.evolves_from_adjacency.transpose(0, 1).matmul(
               evolves_into_weights_->forward(card_embeddings)) +
           adjacency_matrices.pre_evolutions_adjacency.matmul(evolved_from_weights_->forward(card_embeddings)) +
           adjacency_matrices.pre_evolutions_adjacency.transpose(0, 1).matmul(
               evolved_into_weights_->forward(card_embeddings)) +
           adjacency_matrices.attached_energy_adjacency.matmul(
               attached_energy_cards_weights_->forward(card_embeddings)) +
           adjacency_matrices.attached_energy_adjacency.transpose(0, 1).matmul(
               energy_attached_to_weights_->forward(card_embeddings));
}