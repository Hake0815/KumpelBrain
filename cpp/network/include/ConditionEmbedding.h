#ifndef CONDITION_EMBEDDING_H
#define CONDITION_EMBEDDING_H

#include "../include/InstructionDataEmbedding.h"
#include "../include/MultiHeadAttention.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include "../src/serialization/gamecore_serialization.pb.h"

using ProtoBufCondition = gamecore::serialization::ProtoBufCondition;

struct ConditionEmbeddingImpl : torch::nn::Module, SaveLoadMixin<ConditionEmbeddingImpl> {
    ConditionEmbeddingImpl(std::shared_ptr<InstructionDataEmbeddingImpl> instruction_data_embedding,
                           std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out,
                           torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

    std::vector<torch::Tensor> forward(const std::vector<std::vector<ProtoBufCondition>>& conditions_batch);

    std::vector<torch::Tensor> forward_flattened(const nesting::FlattenInstructionsResult& flat, int64_t batch_size);

   private:
    int64_t dimension_out_;
    torch::Device device_;
    torch::Dtype dtype_;

    InstructionDataEmbedding instruction_data_embedding_{nullptr};
    torch::nn::Embedding condition_type_embedding_{nullptr};
    MultiHeadAttention data_multi_head_attention_{nullptr};
    PositionalEmbedding position_embedding_{nullptr};
    MultiHeadAttention conditions_multi_head_attention_{nullptr};

    torch::Tensor compute_data_tensors(const nesting::FlattenInstructionsResult& flat);

    torch::Tensor compute_condition_embeddings(const torch::Tensor& condition_indices,
                                               const torch::Tensor& instruction_data_parent_rows,
                                               const torch::Tensor& embedded_condition_types,
                                               const torch::Tensor& embedded_instruction_data);
};

TORCH_MODULE(ConditionEmbedding);

#endif
