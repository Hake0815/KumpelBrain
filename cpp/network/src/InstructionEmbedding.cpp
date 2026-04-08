#include "../include/InstructionEmbedding.h"

#include "../include/AttentionUtils.h"
#include "../include/Nesting.h"
#include "../include/TensorUtils.h"

namespace serialization = gamecore::serialization;

InstructionEmbeddingImpl::InstructionEmbeddingImpl(
    std::shared_ptr<InstructionDataEmbeddingImpl> instruction_data_embedding,
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out, torch::Device device,
    torch::Dtype dtype)
    : instruction_data_embedding_(instruction_data_embedding),
      dimension_out_(dimension_out),
      device_(device),
      dtype_(dtype) {
    instruction_type_embedding_ =
        register_module("instruction_type_embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(8, dimension_out_).padding_idx(0)));
    data_multi_head_attention_ =
        register_module("data_multi_head_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0, true, device_, dtype_));
    position_embedding_ = shared_embedding_holder->position_embedding_;
    to(device_, dtype_);
}
std::pair<torch::Tensor, torch::Tensor> InstructionEmbeddingImpl::forward(
    const std::vector<std::vector<serialization::ProtoBufInstruction>>& instructions_batch) {
    const int64_t batch_size = static_cast<int64_t>(instructions_batch.size());

    auto flat = nesting::flatten_instructions(instructions_batch, std::nullopt, torch::kInt64);
    flat = nesting::move_flattened_result_to_device(flat, device_);
    return forward_flattened(flat, batch_size);
}

std::pair<torch::Tensor, torch::Tensor> InstructionEmbeddingImpl::forward_flattened(
    const nesting::FlattenInstructionsResult& flat, int64_t batch_size) {
    auto options = torch::TensorOptions().device(device_).dtype(dtype_);
    auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    if (batch_size <= 0) {
        return {torch::empty({0, 0, dimension_out_}, options), torch::empty({0, 0}, mask_options)};
    }
    if (flat.instruction_indices.size(0) == 0) {
        auto zeros_off = torch::zeros({batch_size + 1}, torch::TensorOptions().device(device_).dtype(torch::kInt64));
        auto empty_rows = torch::empty({0, dimension_out_}, options);
        return tensor_utils::pad_by_offsets(empty_rows, zeros_off, dimension_out_,
                                            flat.max_sequence_length_per_batch_item);
    }

    auto embedded_instruction_types = instruction_type_embedding_(flat.instruction_types.to(torch::kLong));

    auto embedded_instruction_data = instruction_data_embedding_->forward(flat).to(device_);

    auto instruction_embeddings =
        compute_instruction_embeddings(flat.instruction_indices, flat.instruction_data_parent_rows,
                                       embedded_instruction_types, embedded_instruction_data,
                                       flat.max_sequence_length_per_parent_row);

    auto batch_offsets = tensor_utils::build_contiguous_offsets(flat.instruction_indices.select(1, 0), batch_size);

    const auto num_instructions = instruction_embeddings.size(0);
    auto local_pos = tensor_utils::local_positions_from_batch_offsets(batch_offsets, num_instructions);
    auto positioned_flat = position_embedding_->forward_packed(instruction_embeddings, local_pos);
    auto [padded_batch, valid_token_mask] = tensor_utils::pad_by_offsets(
        positioned_flat, batch_offsets, dimension_out_, flat.max_sequence_length_per_batch_item);
    return {padded_batch, valid_token_mask};
}

torch::Tensor InstructionEmbeddingImpl::compute_instruction_embeddings(
    const torch::Tensor& instruction_indices, const torch::Tensor& instruction_data_parent_rows,
    const torch::Tensor& embedded_instruction_types, const torch::Tensor& embedded_instruction_data,
    int64_t max_data_sequence_length_per_parent) {
    const auto num_instructions = instruction_indices.size(0);

    if (num_instructions == 0) {
        return torch::empty({0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
    }

    auto data_offsets = tensor_utils::build_parent_offsets(instruction_data_parent_rows, num_instructions);
    auto [padded_data, data_token_mask] = tensor_utils::pad_by_offsets(
        embedded_instruction_data, data_offsets, dimension_out_, max_data_sequence_length_per_parent);
    return attention_utils::query_sum_attention_pooling(
        data_multi_head_attention_, embedded_instruction_types.unsqueeze(1), padded_data, data_token_mask);
}
