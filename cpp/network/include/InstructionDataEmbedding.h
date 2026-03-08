#ifndef INSTRUCTION_DATA_EMBEDDING_H
#define INSTRUCTION_DATA_EMBEDDING_H

#include "../include/AttackDataEmbedding.h"
#include "../include/CardAmountDataEmbedding.h"
#include "../include/DiscardDataEmbedding.h"
#include "../include/FilterEmbedding.h"
#include "../include/PlayerTargetDataEmbedding.h"
#include "../include/ReturnToDeckTypeDataEmbedding.h"
#include "../include/SaveLoadMixin.h"
#include "../include/SharedEmbeddingHolder.h"
#include <array>
#include <tuple>
#include <vector>

struct InstructionDataEmbeddingImpl
    : torch::nn::Module, SaveLoadMixin<InstructionDataEmbeddingImpl> {
  InstructionDataEmbeddingImpl(
      std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
      int64_t dimension_out, torch::Device device = torch::kCPU,
      torch::Dtype dtype = torch::kFloat);

  torch::Tensor forward(
      const torch::Tensor &instruction_indices,
      const torch::Tensor &instruction_data_types,
      const torch::Tensor &instruction_data_type_indices,
      const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
      const std::vector<std::vector<nesting::FilterNode>> &filter_data,
      const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
          &instruction_data_indices,
      int64_t batch_size);

private:
  torch::Tensor sort_tensors_with_respect_to_index(
      const std::array<torch::Tensor, 6> &tensors,
      const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
          &indices) const;

  int64_t dimension_out_;
  torch::Device device_;
  torch::Dtype dtype_;

  AttackDataEmbedding attack_data_embedding_{nullptr};
  DiscardDataEmbedding discard_data_embedding_{nullptr};
  CardAmountDataEmbedding card_amount_data_embedding_{nullptr};
  ReturnToDeckTypeDataEmbedding return_to_deck_type_data_embedding_{nullptr};
  FilterEmbedding filter_embedding_{nullptr};
  PlayerTargetDataEmbedding player_target_data_embedding_{nullptr};
  torch::nn::Embedding instruction_data_type_embedding_{nullptr};
  PositionalEmbedding position_embedding_{nullptr};
};

TORCH_MODULE(InstructionDataEmbedding);

#endif
