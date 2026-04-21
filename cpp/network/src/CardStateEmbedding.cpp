#include "../include/CardStateEmbedding.h"

namespace {

torch::Tensor relational_message(const torch::Tensor& adjacency, torch::nn::Linear& relation_weights,
                                 const torch::Tensor& node_embeddings) {
    return torch::matmul(adjacency, relation_weights(node_embeddings));
}

}  // namespace

CardStateEmbeddingImpl::CardStateEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    card_embedding_ = register_module("card_embedding", CardEmbedding(dimension_out, device, dtype));
    self_loop_weights_ = register_module("self_loop_weights", torch::nn::Linear(dimension_out, dimension_out));
    evolves_from_weights_ = register_module(
        "evolves_from_weights", torch::nn::Linear(torch::nn::LinearOptions(dimension_out, dimension_out).bias(false)));
    evolves_into_weights_ = register_module(
        "evolves_into_weights", torch::nn::Linear(torch::nn::LinearOptions(dimension_out, dimension_out).bias(false)));
    evolved_from_weights_ = register_module(
        "pre_evolution_weights", torch::nn::Linear(torch::nn::LinearOptions(dimension_out, dimension_out).bias(false)));
    evolved_into_weights_ =
        register_module("post_evolution_weights",
                        torch::nn::Linear(torch::nn::LinearOptions(dimension_out, dimension_out).bias(false)));
    energy_attached_to_weights_ =
        register_module("attached_energy_weights",
                        torch::nn::Linear(torch::nn::LinearOptions(dimension_out, dimension_out).bias(false)));
    attached_energy_cards_weights_ = register_module(
        "energy_host_weights", torch::nn::Linear(torch::nn::LinearOptions(dimension_out, dimension_out).bias(false)));
    to(device, dtype);
}

torch::Tensor CardStateEmbeddingImpl::forward(const std::vector<ProtoBufCardState>& card_state_batch) {
    if (card_state_batch.empty()) {
        return torch::empty({0, dimension_out_}, torch::TensorOptions().dtype(dtype_).device(device_));
    }
    std::vector<ProtoBufCard> cards;
    cards.reserve(card_state_batch.size());
    for (const auto& card_state : card_state_batch) {
        cards.push_back(card_state.card());
    }
    auto [embedded_cards, adj] = card_embedding_->forward(cards);

    auto aggregate =
        self_loop_weights_(embedded_cards) +
        relational_message(adj.evolves_from_adjacency, evolves_from_weights_, embedded_cards) +
        relational_message(adj.evolves_from_adjacency.transpose(0, 1), evolves_into_weights_, embedded_cards) +
        relational_message(adj.pre_evolutions_adjacency, evolved_from_weights_, embedded_cards) +
        relational_message(adj.pre_evolutions_adjacency.transpose(0, 1), evolved_into_weights_, embedded_cards) +
        relational_message(adj.attached_energy_adjacency, attached_energy_cards_weights_, embedded_cards) +
        relational_message(adj.attached_energy_adjacency.transpose(0, 1), energy_attached_to_weights_, embedded_cards);

    return torch::relu(aggregate);
}
