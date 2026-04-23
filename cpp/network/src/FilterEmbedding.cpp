#include "network/include/FilterEmbedding.h"

#include <algorithm>

#include "network/include/AttentionUtils.h"
#include "network/include/Nesting.h"
#include "network/include/SharedConstants.h"
namespace serialization = gamecore::serialization;

namespace {

bool has_filter_roots(const nesting::FilterBatchTensors& filter_batch) {
    return filter_batch.root_node_index.defined() && filter_batch.root_node_index.numel() > 0;
}

bool has_leaf_nodes(const nesting::FilterBatchTensors& filter_batch) {
    return filter_batch.leaf_node_index.defined() && filter_batch.leaf_node_index.numel() > 0;
}

torch::Tensor empty_filter_embeddings(torch::Device device, torch::Dtype dtype, int64_t dimension_out) {
    return torch::zeros({0, dimension_out}, torch::TensorOptions().device(device).dtype(dtype));
}

}  // namespace

FilterEmbeddingImpl::FilterEmbeddingImpl(std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
                                         int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    logical_operator_embedding_ = register_module(
        "logical_operator_embedding",
        torch::nn::Embedding(
            torch::nn::EmbeddingOptions(NUMBER_FILTER_LOGICAL_OPERATORS, dimension_out_).padding_idx(0)));
    filter_condition_embedding_ =
        register_module("filter_condition_embedding",
                        FilterConditionEmbedding(shared_embedding_holder, dimension_out_, device_, dtype_));
    multi_head_attention_ =
        register_module("multi_head_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 1), 2, 0.0, false, device_, dtype_));
    to(device_, dtype_);
}

torch::Tensor FilterEmbeddingImpl::forward(const std::vector<serialization::ProtoBufFilter>& filter) {
    if (filter.empty()) {
        return torch::zeros({dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
    }
    return forward_batch(nesting::compile_filter_batch({filter}, device_, torch::kLong));
}

torch::Tensor FilterEmbeddingImpl::forward_batch(const nesting::FilterBatchTensors& filter_batch) {
    if (!has_filter_roots(filter_batch)) {
        return empty_filter_embeddings(device_, dtype_, dimension_out_);
    }

    const auto num_nodes = filter_batch.node_is_leaf.size(0);
    auto node_embeddings =
        torch::zeros({num_nodes, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));

    if (has_leaf_nodes(filter_batch)) {
        auto embedded_conditions = filter_condition_embedding_->forward(
            filter_batch.leaf_field, filter_batch.leaf_compare_op, filter_batch.leaf_value);
        node_embeddings.index_copy_(0, filter_batch.leaf_node_index, embedded_conditions);
    }

    auto internal_mask = torch::logical_not(filter_batch.node_is_leaf);
    if (internal_mask.any().item<bool>()) {
        const auto max_depth = filter_batch.node_depth.max().item<int64_t>();
        for (int64_t depth = max_depth; depth >= 0; --depth) {
            auto depth_mask = torch::logical_and(internal_mask, filter_batch.node_depth.eq(depth));
            auto depth_nodes = torch::nonzero(depth_mask).squeeze(1);
            if (depth_nodes.numel() == 0) {
                continue;
            }

            auto child_start = filter_batch.child_ptr.index_select(0, depth_nodes);
            auto child_end = filter_batch.child_ptr.index_select(0, depth_nodes + 1);
            auto child_count = child_end - child_start;

            auto passthrough_mask = child_count.eq(1);
            if (passthrough_mask.any().item<bool>()) {
                auto passthrough_nodes = depth_nodes.index({passthrough_mask});
                auto passthrough_start = child_start.index({passthrough_mask});
                auto passthrough_children = filter_batch.child_idx.index_select(0, passthrough_start);
                auto passthrough_embeddings = node_embeddings.index_select(0, passthrough_children);
                node_embeddings.index_copy_(0, passthrough_nodes, passthrough_embeddings);
            }

            auto reduce_mask = child_count.gt(1);
            if (!reduce_mask.any().item<bool>()) {
                continue;
            }

            auto reduce_nodes = depth_nodes.index({reduce_mask});
            auto reduce_start = child_start.index({reduce_mask});
            auto reduce_count = child_count.index({reduce_mask});
            const auto max_children = reduce_count.max().item<int64_t>();
            auto positions = torch::arange(max_children, torch::TensorOptions().device(device_).dtype(torch::kLong));
            auto valid_children = positions.unsqueeze(0) < reduce_count.unsqueeze(1);
            auto gather_positions = reduce_start.unsqueeze(1) + positions.unsqueeze(0);
            auto safe_positions = gather_positions.masked_fill(torch::logical_not(valid_children), 0).reshape(-1);
            auto child_indices =
                filter_batch.child_idx.index_select(0, safe_positions).view({reduce_nodes.size(0), max_children});
            auto child_embeddings = node_embeddings.index_select(0, child_indices.reshape(-1))
                                        .view({reduce_nodes.size(0), max_children, dimension_out_});
            child_embeddings = child_embeddings * valid_children.unsqueeze(-1).to(child_embeddings.dtype());

            auto operator_embeddings =
                logical_operator_embedding_(filter_batch.node_logical_operator.index_select(0, reduce_nodes))
                    .unsqueeze(1);
            auto query = torch::cat({child_embeddings, operator_embeddings}, 1);
            auto valid_token_mask = torch::cat(
                {valid_children,
                 torch::ones({reduce_nodes.size(0), 1}, torch::TensorOptions().device(device_).dtype(torch::kBool))},
                1);
            auto reduced =
                attention_utils::masked_self_attention_reduce(multi_head_attention_, query, valid_token_mask);
            node_embeddings.index_copy_(0, reduce_nodes, reduced);
        }
    }

    return node_embeddings.index_select(0, filter_batch.root_node_index);
}
