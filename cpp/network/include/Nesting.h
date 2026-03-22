#ifndef NESTING_H
#define NESTING_H

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/serialization/gamecore_serialization.pb.h"
#include <torch/torch.h>

namespace nesting {

constexpr size_t kNumInstructionDataTypes = 6;

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

using ReduceCombineFunction =
    std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                std::optional<int64_t>)>;

struct ReduceRequest {
  const std::vector<torch::Tensor> &flattened_input;
  const std::vector<GroupIndex> &groups;
  const OperatorMap &operators;
  const ReduceCombineFunction &combine_function;
};

struct FilterBatchTensors {
  torch::Tensor node_is_leaf{};
  torch::Tensor node_logical_operator{};
  torch::Tensor node_depth{};
  torch::Tensor child_ptr{};
  torch::Tensor child_idx{};
  torch::Tensor leaf_node_index{};
  torch::Tensor leaf_field{};
  torch::Tensor leaf_compare_op{};
  torch::Tensor leaf_value{};
  torch::Tensor root_node_index{};
};

struct FlattenInstructionsResult {
  torch::Tensor instruction_types{};
  torch::Tensor instruction_indices{};
  torch::Tensor instruction_data_types{};
  torch::Tensor instruction_data_parent_rows{};
  torch::Tensor instruction_data_type_indices{};
  torch::Tensor instruction_data_reorder{};
  std::array<torch::Tensor, kNumInstructionDataTypes> instruction_data_tensors{};
  FilterBatchTensors filter_batch{};
};

std::string group_index_key(const GroupIndex &group_index);
bool is_prefix(const GroupIndex &prefix, const GroupIndex &test);

std::vector<TraverseEntry> traverse_filter(
    const std::vector<ProtoBufFilter> &nested_input);
FlattenResult flatten(const std::vector<TraverseEntry> &entries);

std::vector<torch::Tensor> reduce(const ReduceRequest &request);

FlattenInstructionsResult flatten_instructions(
    const std::vector<std::vector<ProtoBufInstruction>> &instructions,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FlattenInstructionsResult flatten_conditions(
    const std::vector<std::vector<ProtoBufCondition>> &conditions,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FilterBatchTensors compile_filter_batch(
    const std::vector<std::vector<ProtoBufFilter>> &filters,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FlattenInstructionsResult
move_flattened_result_to_device(const FlattenInstructionsResult &result,
                                torch::Device device);

FilterBatchTensors move_filter_batch_to_device(const FilterBatchTensors &result,
                                               torch::Device device);

} // namespace nesting

#endif
