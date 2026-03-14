#ifndef NESTING_H
#define NESTING_H

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../src/serialization/gamecore_serialization.pb.h"
#include <torch/torch.h>

namespace nesting {

using GroupIndex = std::vector<int64_t>;
using OperatorMap = std::unordered_map<std::string, int64_t>;

struct TraverseEntry {
  std::vector<int64_t> value{};
  GroupIndex group_index{};
  int64_t op = 0;
};

struct FlattenResult {
  std::vector<std::vector<int64_t>> flattened_input{};
  std::vector<GroupIndex> groups{};
  OperatorMap operators{};
};

struct FlattenInstructionsResult {
  torch::Tensor instruction_types{};
  torch::Tensor instruction_indices{};
  torch::Tensor instruction_data_types{};
  torch::Tensor instruction_data_type_indices{};
  std::array<std::vector<torch::Tensor>, 6> instruction_data{};
  std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
      instruction_data_indices{};
  std::vector<std::vector<gamecore::serialization::ProtoBufFilter>> filter_data{};
};

std::string group_index_key(const GroupIndex &group_index);
bool is_prefix(const GroupIndex &prefix, const GroupIndex &test);

std::vector<TraverseEntry> traverse_filter(
    const std::vector<gamecore::serialization::ProtoBufFilter> &nested_input);
FlattenResult flatten(const std::vector<TraverseEntry> &entries);

std::vector<torch::Tensor> reduce(
    const std::vector<torch::Tensor> &flattened_input,
    const std::vector<GroupIndex> &groups, const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)> &combine_function);

torch::Tensor vectorize_amount_data(
    const gamecore::serialization::ProtoBufCardAmountInstructionData
        &amount_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_attack_data(
    const gamecore::serialization::ProtoBufAttackInstructionData &attack_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_discard_data(
    const gamecore::serialization::ProtoBufDiscardInstructionData &discard_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_return_to_deck_type_data(
    const gamecore::serialization::ProtoBufReturnToDeckTypeInstructionData
        &return_to_deck_type_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_player_target_data(
    const gamecore::serialization::ProtoBufPlayerTargetInstructionData
        &player_target_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

torch::Tensor vectorize_payload(
    const gamecore::serialization::ProtoBufInstructionData &data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FlattenInstructionsResult flatten_instructions(
    const std::vector<std::vector<gamecore::serialization::ProtoBufInstruction>>
        &instructions,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FlattenInstructionsResult flatten_conditions(
    const std::vector<std::vector<gamecore::serialization::ProtoBufCondition>>
        &conditions,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

} // namespace nesting

#endif
