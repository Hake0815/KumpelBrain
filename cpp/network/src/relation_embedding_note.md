# Relation embedding note

no normalization in R-GCN-style aggregation, because counts can matter, could be recovered by code snippet

```cpp
//deg[i] ≈ sum of A[i, j]
torch::Tensor coo_row_degree_clamped(const torch::Tensor& sparse_coo) {
    const auto a = sparse_coo.coalesce();
    const int64_t n = a.size(0);
    const auto fp_dtype = (a.scalar_type() == torch::kFloat64 || a.scalar_type() == torch::kComplexDouble)
                              ? torch::kFloat64
                              : torch::kFloat;
    // Do not use sparse_coo.options() here: it can carry sparse layout and breaks scatter_add_.
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
    const auto deg = coo_row_degree_clamped(adjacency);
    const auto messages = torch::matmul(adjacency, relation_weights(node_embeddings));
    return messages / deg.unsqueeze(-1);
}
```
