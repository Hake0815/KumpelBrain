#include "../include/CardStateEmbedding.h"

namespace {

torch::Tensor coo_row_degree_clamped(const torch::Tensor& sparse_coo) {
    const auto a = sparse_coo.coalesce();
    const int64_t n = a.size(0);
    const auto fp_dtype = (a.scalar_type() == torch::kFloat64 || a.scalar_type() == torch::kComplexDouble)
                              ? torch::kFloat64
                              : torch::kFloat;
    const auto deg_opts = torch::TensorOptions().device(sparse_coo.device()).dtype(fp_dtype);
    if (a._nnz() == 0) {
        return torch::ones({n}, deg_opts);
    }
    const auto idx = a.indices();
    const auto rows = idx[0];
    const auto vals = a.values().to(fp_dtype);
    auto deg = torch::zeros({n}, deg_opts);
    deg.scatter_add_(0, rows, vals);
    return deg.clamp_min(1.0);
}

torch::Tensor relational_message(const torch::Tensor& adjacency, torch::nn::Linear& relation_weights,
                                 const torch::Tensor& node_embeddings) {
    auto deg = coo_row_degree_clamped(adjacency);
    auto messages = torch::matmul(adjacency, relation_weights(node_embeddings));
    return messages / deg.unsqueeze(-1);
}

torch::Tensor sparse_col_sum(const torch::Tensor& sparse_coo, int64_t n, torch::Device device, torch::Dtype dtype) {
    const auto a = sparse_coo.coalesce();
    auto result = torch::zeros({n}, torch::TensorOptions().device(device).dtype(dtype));
    if (a._nnz() == 0) {
        return result;
    }
    const auto cols = a.indices()[1];
    const auto vals = a.values().to(dtype);
    result.scatter_add_(0, cols, vals);
    return result;
}

torch::Tensor sparse_row_sum(const torch::Tensor& sparse_coo, int64_t n, torch::Device device, torch::Dtype dtype) {
    const auto a = sparse_coo.coalesce();
    auto result = torch::zeros({n}, torch::TensorOptions().device(device).dtype(dtype));
    if (a._nnz() == 0) {
        return result;
    }
    const auto rows = a.indices()[0];
    const auto vals = a.values().to(dtype);
    result.scatter_add_(0, rows, vals);
    return result;
}

}  // namespace

CardStateEmbeddingImpl::CardStateEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    shared_embedding_holder_ =
        register_module("shared_embedding_holder", SharedEmbeddingHolder(dimension_out, device, dtype));
    card_embedding_ =
        register_module("card_embedding", CardEmbedding(shared_embedding_holder_.ptr(), dimension_out, device, dtype));
    position_embedding_ = register_module(
        "position_embedding", CardPositionEmbedding(shared_embedding_holder_.ptr(), dimension_out, device, dtype));
    card_position_gate_ = register_module("card_position_gate", torch::nn::Linear(2 * dimension_out, dimension_out));
    degree_count_embedding_ = register_module("degree_count_embedding",
                                              NormalizedLinear(kNumDegreeFeatures, dimension_out, 10.0, device, dtype));

    const auto no_bias = [&](const std::string& name) {
        return register_module(name,
                               torch::nn::Linear(torch::nn::LinearOptions(dimension_out, dimension_out).bias(false)));
    };

    for (int64_t layer = 0; layer < kNumRgcnLayers; ++layer) {
        const auto prefix = "layer_" + std::to_string(layer) + "_";
        auto& w = rgcn_layers_[static_cast<size_t>(layer)];
        w.self_loop = register_module(prefix + "self_loop_weights", torch::nn::Linear(dimension_out, dimension_out));
        w.evolves_from = no_bias(prefix + "evolves_from_weights");
        w.evolves_into = no_bias(prefix + "evolves_into_weights");
        w.evolved_from = no_bias(prefix + "pre_evolution_weights");
        w.evolved_into = no_bias(prefix + "post_evolution_weights");
        w.energy_attached_to = no_bias(prefix + "attached_energy_weights");
        w.attached_energy_cards = no_bias(prefix + "energy_host_weights");
    }

    to(device, dtype);
}

torch::Tensor CardStateEmbeddingImpl::forward(const std::vector<ProtoBufCardState>& card_state_batch) {
    if (card_state_batch.empty()) {
        return torch::empty({0, dimension_out_}, torch::TensorOptions().dtype(dtype_).device(device_));
    }
    auto [embedded_cards, adj] = card_embedding_->forward(card_state_batch);
    auto position_vec = position_embedding_->forward(card_state_batch);

    auto gate = torch::sigmoid(card_position_gate_(torch::cat({embedded_cards, position_vec}, 1)));
    auto h = embedded_cards * (1.0 - gate) + position_vec * gate;

    const int64_t num_cards = static_cast<int64_t>(card_state_batch.size());
    auto degrees = compute_degree_features(adj, num_cards, device_, dtype_);
    h = h + degree_count_embedding_(degrees);

    for (auto& layer_weights : rgcn_layers_) {
        h = torch::relu(aggregate_one_layer(h, adj, layer_weights)) + h;
    }

    return h;
}

torch::Tensor CardStateEmbeddingImpl::aggregate_one_layer(const torch::Tensor& node_emb, const AdjacencyMatrices& adj,
                                                          RgcnLayerWeights& weights) {
    return weights.self_loop(node_emb) +
           relational_message(adj.evolves_from_adjacency, weights.evolves_from, node_emb) +
           relational_message(adj.evolves_from_adjacency.transpose(0, 1), weights.evolves_into, node_emb) +
           relational_message(adj.pre_evolutions_adjacency, weights.evolved_from, node_emb) +
           relational_message(adj.pre_evolutions_adjacency.transpose(0, 1), weights.evolved_into, node_emb) +
           relational_message(adj.attached_energy_adjacency, weights.attached_energy_cards, node_emb) +
           relational_message(adj.attached_energy_adjacency.transpose(0, 1), weights.energy_attached_to, node_emb);
}

torch::Tensor CardStateEmbeddingImpl::compute_degree_features(const AdjacencyMatrices& adj, int64_t num_cards,
                                                              torch::Device device, torch::Dtype dtype) {
    return torch::stack({sparse_row_sum(adj.evolves_from_adjacency, num_cards, device, dtype),
                         sparse_col_sum(adj.evolves_from_adjacency, num_cards, device, dtype),
                         sparse_row_sum(adj.pre_evolutions_adjacency, num_cards, device, dtype),
                         sparse_col_sum(adj.pre_evolutions_adjacency, num_cards, device, dtype),
                         sparse_row_sum(adj.attached_energy_adjacency, num_cards, device, dtype),
                         sparse_col_sum(adj.attached_energy_adjacency, num_cards, device, dtype)},
                        1);
}
