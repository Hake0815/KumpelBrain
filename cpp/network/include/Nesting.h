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

using ProtoBufAttackInstructionData =
    gamecore::serialization::ProtoBufAttackInstructionData;
using ProtoBufCardAmountInstructionData =
    gamecore::serialization::ProtoBufCardAmountInstructionData;
using ProtoBufCondition = gamecore::serialization::ProtoBufCondition;
using ProtoBufDiscardInstructionData =
    gamecore::serialization::ProtoBufDiscardInstructionData;
using ProtoBufFilter = gamecore::serialization::ProtoBufFilter;
using ProtoBufInstruction = gamecore::serialization::ProtoBufInstruction;
using ProtoBufInstructionData = gamecore::serialization::ProtoBufInstructionData;
using ProtoBufPlayerTargetInstructionData =
    gamecore::serialization::ProtoBufPlayerTargetInstructionData;
using ProtoBufReturnToDeckTypeInstructionData =
    gamecore::serialization::ProtoBufReturnToDeckTypeInstructionData;

using GroupIndex = std::vector<int64_t>;

struct GroupIndexHash {
  size_t operator()(const GroupIndex &group_index) const noexcept {
    size_t seed = 0;
    for (const auto value : group_index) {
      seed ^= std::hash<int64_t>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

using OperatorMap = std::unordered_map<GroupIndex, int64_t, GroupIndexHash>;

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
  torch::Tensor instruction_data_parent_rows{};
  torch::Tensor instruction_data_type_indices{};
  std::array<std::vector<torch::Tensor>, 6> instruction_data{};
  std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
      instruction_data_indices{};
  std::vector<std::vector<ProtoBufFilter>> filter_data{};
};

std::string group_index_key(const GroupIndex &group_index);
bool is_prefix(const GroupIndex &prefix, const GroupIndex &test);

std::vector<TraverseEntry> traverse_filter(
    const std::vector<ProtoBufFilter> &nested_input);
FlattenResult flatten(const std::vector<TraverseEntry> &entries);

std::vector<torch::Tensor> reduce(
    const std::vector<torch::Tensor> &flattened_input,
    const std::vector<GroupIndex> &groups, const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)> &combine_function);

torch::Tensor vectorize_amount_data(
    const ProtoBufCardAmountInstructionData &amount_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_attack_data(
    const ProtoBufAttackInstructionData &attack_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_discard_data(
    const ProtoBufDiscardInstructionData &discard_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_return_to_deck_type_data(
    const ProtoBufReturnToDeckTypeInstructionData &return_to_deck_type_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_player_target_data(
    const ProtoBufPlayerTargetInstructionData &player_target_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

torch::Tensor vectorize_payload(
    const ProtoBufInstructionData &data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FlattenInstructionsResult flatten_instructions(
    const std::vector<std::vector<ProtoBufInstruction>> &instructions,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FlattenInstructionsResult flatten_conditions(
    const std::vector<std::vector<ProtoBufCondition>> &conditions,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

} // namespace nesting

#endif
