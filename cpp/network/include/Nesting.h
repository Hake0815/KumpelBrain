#ifndef NESTING_H
#define NESTING_H

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <torch/torch.h>

namespace nesting {

using GroupIndex = std::vector<int64_t>;
using OperatorMap = std::unordered_map<std::string, int64_t>;

struct Condition {
  int64_t field = 0;
  int64_t operation = 0;
  int64_t value = 0;
};

struct FilterNode {
  bool is_leaf = true;
  Condition condition{};
  int64_t logical_operator = 0;
  std::vector<FilterNode> operands{};
};

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

struct AmountData {
  int64_t min = 0;
  int64_t max = 0;
  int64_t from_position = 0;
};

struct AttackData {
  int64_t attack_target = 0;
  int64_t damage = 0;
};

struct DiscardData {
  int64_t target_source = 0;
};

struct ReturnToDeckTypeData {
  int64_t return_to_deck_type = 0;
  int64_t from_position = 0;
};

struct PlayerTargetData {
  int64_t player_target = 0;
};

using PayloadVariant = std::variant<AttackData, DiscardData, AmountData,
                                    ReturnToDeckTypeData, PlayerTargetData>;

struct InstructionData {
  int64_t instruction_data_type = 0;
  std::optional<PayloadVariant> payload{};
  std::optional<std::vector<FilterNode>> filter_payload{};
};

struct Instruction {
  int64_t instruction_type = 0;
  int64_t condition_type = 0;
  std::vector<InstructionData> data{};
};

struct FlattenInstructionsResult {
  torch::Tensor instruction_types{};
  torch::Tensor instruction_indices{};
  torch::Tensor instruction_data_types{};
  torch::Tensor instruction_data_type_indices{};
  std::array<std::vector<torch::Tensor>, 6> instruction_data{};
  std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
      instruction_data_indices{};
  std::vector<std::vector<FilterNode>> filter_data{};
};

std::string group_index_key(const GroupIndex &group_index);
bool is_prefix(const GroupIndex &prefix, const GroupIndex &test);

std::vector<TraverseEntry> traverse_filter(const std::vector<FilterNode> &nested_input);
FlattenResult flatten(const std::vector<TraverseEntry> &entries);

std::vector<torch::Tensor> reduce(
    const std::vector<torch::Tensor> &flattened_input,
    const std::vector<GroupIndex> &groups, const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)> &combine_function);

torch::Tensor vectorize_amount_data(
    const AmountData &amount_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_attack_data(
    const AttackData &attack_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_discard_data(
    const DiscardData &discard_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_return_to_deck_type_data(
    const ReturnToDeckTypeData &return_to_deck_type_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);
torch::Tensor vectorize_player_target_data(
    const PlayerTargetData &player_target_data,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

torch::Tensor vectorize_payload(
    const PayloadVariant &payload, int64_t data_type,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

FlattenInstructionsResult flatten_instructions(
    const std::string &type_key,
    const std::vector<std::vector<Instruction>> &instructions,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<torch::Dtype> dtype = std::nullopt);

} // namespace nesting

#endif
