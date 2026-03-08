#include "../include/InstructionDataEmbedding.h"
#include <algorithm>

InstructionDataEmbeddingImpl::InstructionDataEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder,
    int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
  attack_data_embedding_ =
      register_module("attack_data_embedding",
                      AttackDataEmbedding(dimension_out_, device_, dtype_));
  discard_data_embedding_ =
      register_module("discard_data_embedding",
                      DiscardDataEmbedding(dimension_out_, device_, dtype_));
  card_amount_data_embedding_ =
      register_module("card_amount_data_embedding",
                      CardAmountDataEmbedding(shared_embedding_holder,
                                              dimension_out_, device_, dtype_));
  return_to_deck_type_data_embedding_ = register_module(
      "return_to_deck_type_data_embedding",
      ReturnToDeckTypeDataEmbedding(shared_embedding_holder, dimension_out_,
                                    device_, dtype_));
  filter_embedding_ = register_module(
      "filter_embedding", FilterEmbedding(shared_embedding_holder,
                                          dimension_out_, device_, dtype_));
  player_target_data_embedding_ = register_module(
      "player_target_data_embedding",
      PlayerTargetDataEmbedding(shared_embedding_holder, dimension_out_,
                                device_, dtype_));
  instruction_data_type_embedding_ = register_module(
      "instruction_data_type_embedding",
      torch::nn::Embedding(
          torch::nn::EmbeddingOptions(6, dimension_out_).padding_idx(0)));
  position_embedding_ = shared_embedding_holder->position_embedding_;

  to(device_, dtype_);
}

torch::Tensor InstructionDataEmbeddingImpl::sort_tensors_with_respect_to_index(
    const std::array<torch::Tensor, 6> &tensors,
    const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
        &indices) const {
  std::vector<std::pair<std::tuple<int64_t, int64_t, int64_t>, torch::Tensor>>
      indexed_tensors;

  for (size_t group = 0; group < tensors.size(); ++group) {
    if (!tensors[group].defined() || tensors[group].numel() == 0) {
      continue;
    }

    const auto rows = tensors[group].size(0);
    if (static_cast<int64_t>(indices[group].size()) != rows) {
      throw std::invalid_argument("Tensor/index count mismatch in sorting");
    }

    for (int64_t i = 0; i < rows; ++i) {
      indexed_tensors.emplace_back(indices[group][i], tensors[group][i]);
    }
  }

  std::sort(indexed_tensors.begin(), indexed_tensors.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<torch::Tensor> sorted;
  sorted.reserve(indexed_tensors.size());
  for (const auto &item : indexed_tensors) {
    sorted.push_back(item.second);
  }

  if (sorted.empty()) {
    return torch::empty({0, dimension_out_},
                        torch::TensorOptions().device(device_).dtype(dtype_));
  }
  return torch::stack(sorted, 0);
}

torch::Tensor InstructionDataEmbeddingImpl::forward(
    const torch::Tensor &instruction_indices,
    const torch::Tensor &instruction_data_types,
    const torch::Tensor &instruction_data_type_indices,
    const std::array<std::vector<torch::Tensor>, 6> &instruction_data,
    const std::vector<std::vector<nesting::FilterNode>> &filter_data,
    const std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
        &instruction_data_indices,
    int64_t batch_size) {
  (void)instruction_indices;
  (void)instruction_data_type_indices;
  (void)batch_size;

  auto instruction_data_type_embeddings =
      instruction_data_type_embedding_(instruction_data_types.to(torch::kLong));

  std::array<torch::Tensor, 6> embedded_data;

  if (!instruction_data[0].empty()) {
    embedded_data[0] = attack_data_embedding_->forward(
        torch::stack(instruction_data[0], 0).to(device_));
  }
  if (!instruction_data[1].empty()) {
    auto discard_tensor = torch::stack(instruction_data[1], 0).to(device_);
    if (discard_tensor.dim() > 1 && discard_tensor.size(1) == 1) {
      discard_tensor = discard_tensor.squeeze(1);
    }
    embedded_data[1] =
        discard_data_embedding_->forward(discard_tensor.to(torch::kLong));
  }
  if (!instruction_data[2].empty()) {
    embedded_data[2] = card_amount_data_embedding_->forward(
        torch::stack(instruction_data[2], 0).to(device_));
  }
  if (!instruction_data[3].empty()) {
    embedded_data[3] = return_to_deck_type_data_embedding_->forward(
        torch::stack(instruction_data[3], 0).to(device_));
  }
  if (!filter_data.empty()) {
    std::vector<torch::Tensor> filter_embeddings;
    filter_embeddings.reserve(filter_data.size());
    for (const auto &single_filter : filter_data) {
      auto emb = filter_embedding_->forward(single_filter);
      if (emb.dim() == 2 && emb.size(0) == 1) {
        emb = emb.squeeze(0);
      }
      filter_embeddings.push_back(emb);
    }
    if (!filter_embeddings.empty()) {
      embedded_data[4] = torch::stack(filter_embeddings, 0);
    }
  }
  if (!instruction_data[5].empty()) {
    auto player_tensor = torch::stack(instruction_data[5], 0).to(device_);
    if (player_tensor.dim() > 1 && player_tensor.size(1) == 1) {
      player_tensor = player_tensor.squeeze(1);
    }
    embedded_data[5] =
        player_target_data_embedding_->forward(player_tensor.to(torch::kLong));
  }

  auto sorted_data = sort_tensors_with_respect_to_index(
      embedded_data, instruction_data_indices);
  return sorted_data + instruction_data_type_embeddings;
}
