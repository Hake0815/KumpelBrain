#ifndef CONDITION_EMBEDDING_H
#define CONDITION_EMBEDDING_H

#include "../include/InstructionDataEmbedding.h"
#include "../include/MultiHeadAttention.h"
#include "../include/Nesting.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"

struct ConditionEmbeddingImpl : torch::nn::Module,
                                SaveLoadMixin<ConditionEmbeddingImpl> {
  ConditionEmbeddingImpl(
      std::shared_ptr<InstructionDataEmbeddingImpl> instruction_data_embedding,
      std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
      int64_t dimension_out, torch::Device device = torch::kCPU,
      torch::Dtype dtype = torch::kFloat);

  torch::Tensor
  forward(const std::vector<std::vector<gamecore::serialization::ProtoBufCondition>>
              &conditions_batch);

private:
  torch::Tensor compute_data_tensors(
      const torch::Tensor &condition_indices,
      const torch::Tensor &instruction_data_types,
      const torch::Tensor &instruction_data_type_indices,
      const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
      const std::vector<
          std::vector<gamecore::serialization::ProtoBufFilter>> &filter_data,
      const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
          &instruction_data_indices,
      int64_t batch_size);

  torch::Tensor compute_condition_embeddings(
      const torch::Tensor &condition_types, const torch::Tensor &condition_indices,
      const torch::Tensor &instruction_data_type_indices,
      const torch::Tensor &data_tensors);

  int64_t dimension_out_;
  torch::Device device_;
  torch::Dtype dtype_;

  InstructionDataEmbedding instruction_data_embedding_{nullptr};
  torch::nn::Embedding condition_type_embedding_{nullptr};
  MultiHeadAttention data_multi_head_attention_{nullptr};
  PositionalEmbedding position_embedding_{nullptr};
  MultiHeadAttention conditions_multi_head_attention_{nullptr};
};

TORCH_MODULE(ConditionEmbedding);

#endif
