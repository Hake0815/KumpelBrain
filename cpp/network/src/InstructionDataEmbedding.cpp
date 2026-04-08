#include "../include/InstructionDataEmbedding.h"

namespace {

bool has_rows(const torch::Tensor& tensor) { return tensor.defined() && tensor.numel() > 0; }

torch::Tensor squeeze_single_feature_column(const torch::Tensor& tensor) {
    if (tensor.dim() > 1 && tensor.size(1) == 1) {
        return tensor.squeeze(1);
    }
    return tensor;
}

void append_if_present(std::vector<torch::Tensor>& parts, const torch::Tensor& tensor) {
    if (has_rows(tensor)) {
        parts.push_back(tensor);
    }
}

}  // namespace

InstructionDataEmbeddingImpl::InstructionDataEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out, torch::Device device,
    torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    attack_data_embedding_ = register_module(
        "attack_data_embedding", AttackDataEmbedding(shared_embedding_holder, dimension_out_, device_, dtype_));
    discard_data_embedding_ =
        register_module("discard_data_embedding", DiscardDataEmbedding(dimension_out_, device_, dtype_));
    card_amount_data_embedding_ =
        register_module("card_amount_data_embedding",
                        CardAmountDataEmbedding(shared_embedding_holder, dimension_out_, device_, dtype_));
    return_to_deck_type_data_embedding_ =
        register_module("return_to_deck_type_data_embedding",
                        ReturnToDeckTypeDataEmbedding(shared_embedding_holder, dimension_out_, device_, dtype_));
    filter_embedding_ =
        register_module("filter_embedding", FilterEmbedding(shared_embedding_holder, dimension_out_, device_, dtype_));
    player_target_data_embedding_ =
        register_module("player_target_data_embedding",
                        PlayerTargetDataEmbedding(shared_embedding_holder, dimension_out_, device_, dtype_));
    instruction_data_type_embedding_ =
        register_module("instruction_data_type_embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(6, dimension_out_).padding_idx(0)));
    position_embedding_ = shared_embedding_holder->position_embedding_;

    to(device_, dtype_);
}

torch::Tensor InstructionDataEmbeddingImpl::forward(const nesting::FlattenInstructionsResult& flat) {
    return embed_dense_payloads(flat.instruction_data_types, flat.instruction_data_tensors, flat.filter_batch,
                                flat.instruction_data_reorder);
}

torch::Tensor InstructionDataEmbeddingImpl::embed_dense_payloads(
    const torch::Tensor& instruction_data_types,
    const std::array<torch::Tensor, nesting::kNumInstructionDataTypes>& instruction_data_tensors,
    const nesting::FilterBatchTensors& filter_batch, const torch::Tensor& instruction_data_reorder) {
    if (instruction_data_types.numel() == 0) {
        return torch::empty({0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
    }

    auto instruction_data_type_embeddings = instruction_data_type_embedding_(instruction_data_types.to(torch::kLong));

    std::array<torch::Tensor, nesting::kNumInstructionDataTypes> embedded_data;

    if (has_rows(instruction_data_tensors[0])) {
        embedded_data[0] = attack_data_embedding_->forward(instruction_data_tensors[0]);
    }
    if (has_rows(instruction_data_tensors[1])) {
        auto discard_tensor = squeeze_single_feature_column(instruction_data_tensors[1]);
        embedded_data[1] = discard_data_embedding_->forward(discard_tensor.to(torch::kLong));
    }
    if (has_rows(instruction_data_tensors[2])) {
        embedded_data[2] = card_amount_data_embedding_->forward(instruction_data_tensors[2]);
    }
    if (has_rows(instruction_data_tensors[3])) {
        embedded_data[3] = return_to_deck_type_data_embedding_->forward(instruction_data_tensors[3]);
    }
    if (has_rows(filter_batch.root_node_index)) {
        embedded_data[4] = filter_embedding_->forward_batch(filter_batch);
    }
    if (has_rows(instruction_data_tensors[5])) {
        auto player_tensor = squeeze_single_feature_column(instruction_data_tensors[5]);
        embedded_data[5] = player_target_data_embedding_->forward(player_tensor.to(torch::kLong));
    }

    std::vector<torch::Tensor> concatenated_parts;
    concatenated_parts.reserve(embedded_data.size());
    for (const auto& tensor : embedded_data) {
        append_if_present(concatenated_parts, tensor);
    }
    if (concatenated_parts.empty()) {
        return instruction_data_type_embeddings;
    }

    auto concatenated = torch::cat(concatenated_parts, 0);
    // Filters (type 4) have width 0 in the payload tensor but still occupy
    // positions in the global reorder mapping so that their embeddings are
    // interleaved with the other data-type embeddings in the original order.
    auto sorted_data = instruction_data_reorder.numel() == 0
                           ? concatenated
                           : concatenated.index_select(0, instruction_data_reorder.to(torch::kLong));
    return sorted_data + instruction_data_type_embeddings;
}
