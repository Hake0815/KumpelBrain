#include "../include/Nesting.h"
#include "../include/TensorUtils.h"

#include <sstream>
#include <stdexcept>

namespace nesting {

namespace {

torch::TensorOptions make_options(std::optional<torch::Device> device,
                                  std::optional<torch::Dtype> dtype) {
  auto options = torch::TensorOptions();
  if (device.has_value()) {
    options = options.device(*device);
  }
  if (dtype.has_value()) {
    options = options.dtype(*dtype);
  }
  return options;
}

std::optional<int64_t> get_operator(const OperatorMap &operators,
                                    const GroupIndex &group_index) {
  const auto key = group_index_key(group_index);
  auto it = operators.find(key);
  if (it == operators.end()) {
    return std::nullopt;
  }
  return it->second;
}

void traverse_filter_node(const std::vector<FilterNode> &nested_input,
                          GroupIndex &path_list, int64_t op,
                          std::vector<TraverseEntry> &out) {
  for (size_t i = 0; i < nested_input.size(); ++i) {
    const auto &node = nested_input[i];
    if (node.is_leaf) {
      out.push_back(TraverseEntry{
          {node.condition.field, node.condition.operation, node.condition.value},
          path_list, op});
    } else {
      path_list.push_back(static_cast<int64_t>(i));
      traverse_filter_node(node.operands, path_list, node.logical_operator, out);
      path_list.pop_back();
    }
  }
}

void break_down_stack(
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const GroupIndex &group_index,
    const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)> &combine_function);

void add_to_stack(
    const torch::Tensor &element,
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const GroupIndex &group_index,
    const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)> &combine_function) {
  if (!current_groups.empty() && group_index == current_groups.back()) {
    current_combination.back().push_back(element);
    return;
  }

  break_down_stack(current_combination, current_groups, group_index, operators,
                   combine_function);

  if (!current_groups.empty() && group_index == current_groups.back()) {
    current_combination.back().push_back(element);
  } else {
    current_combination.push_back({element});
    current_groups.push_back(group_index);
  }
}

void break_down_stack(
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const GroupIndex &group_index,
    const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)> &combine_function) {
  while (!current_groups.empty() &&
         !is_prefix(current_groups.back(), group_index)) {
    const auto current_group = current_groups.back();
    current_groups.pop_back();

    const auto combined =
        combine_function(current_combination.back(),
                         get_operator(operators, current_group));
    current_combination.pop_back();

    GroupIndex reduced_index = current_group;
    if (!reduced_index.empty()) {
      reduced_index.pop_back();
    }
    add_to_stack(combined, current_combination, current_groups, reduced_index,
                 operators, combine_function);
  }
}

} // namespace

std::string group_index_key(const GroupIndex &group_index) {
  std::ostringstream oss;
  for (size_t i = 0; i < group_index.size(); ++i) {
    if (i > 0) {
      oss << ".";
    }
    oss << group_index[i];
  }
  return oss.str();
}

bool is_prefix(const GroupIndex &prefix, const GroupIndex &test) {
  if (prefix.size() > test.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i] != test[i]) {
      return false;
    }
  }
  return true;
}

std::vector<TraverseEntry>
traverse_filter(const std::vector<FilterNode> &nested_input) {
  std::vector<TraverseEntry> out;
  GroupIndex path_list;

  for (size_t i = 0; i < nested_input.size(); ++i) {
    const auto &node = nested_input[i];
    path_list.push_back(static_cast<int64_t>(i));
    if (node.is_leaf) {
      out.push_back(TraverseEntry{
          {node.condition.field, node.condition.operation, node.condition.value},
          path_list, 0});
    } else {
      traverse_filter_node(node.operands, path_list, node.logical_operator, out);
    }
    path_list.pop_back();
  }

  return out;
}

FlattenResult flatten(const std::vector<TraverseEntry> &entries) {
  FlattenResult result;
  result.flattened_input.reserve(entries.size());
  result.groups.reserve(entries.size());

  for (const auto &entry : entries) {
    result.flattened_input.push_back(entry.value);
    result.groups.push_back(entry.group_index);
    result.operators[group_index_key(entry.group_index)] = entry.op;
  }
  return result;
}

std::vector<torch::Tensor> reduce(
    const std::vector<torch::Tensor> &flattened_input,
    const std::vector<GroupIndex> &groups, const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)> &combine_function) {
  std::vector<std::vector<torch::Tensor>> current_combination;
  std::vector<GroupIndex> current_groups;
  int64_t current_batch_index = 0;

  for (size_t i = 0; i < groups.size(); ++i) {
    if (current_groups.empty()) {
      current_groups.push_back(GroupIndex{current_batch_index});
      current_batch_index += 1;
      current_combination.push_back({});
    }

    add_to_stack(flattened_input[i], current_combination, current_groups,
                 groups[i], operators, combine_function);
  }

  break_down_stack(current_combination, current_groups, GroupIndex{}, operators,
                   combine_function);
  if (current_combination.empty()) {
    return {};
  }
  return current_combination.back();
}

torch::Tensor vectorize_amount_data(const AmountData &amount_data,
                                    std::optional<torch::Device> device,
                                    std::optional<torch::Dtype> dtype) {
  return torch::tensor({amount_data.min, amount_data.max, amount_data.from_position},
                       make_options(device, dtype));
}

torch::Tensor vectorize_attack_data(const AttackData &attack_data,
                                    std::optional<torch::Device> device,
                                    std::optional<torch::Dtype> dtype) {
  return torch::tensor({attack_data.attack_target, attack_data.damage},
                       make_options(device, dtype));
}

torch::Tensor vectorize_discard_data(const DiscardData &discard_data,
                                     std::optional<torch::Device> device,
                                     std::optional<torch::Dtype> dtype) {
  return torch::tensor({discard_data.target_source}, make_options(device, dtype));
}

torch::Tensor vectorize_return_to_deck_type_data(
    const ReturnToDeckTypeData &return_to_deck_type_data,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return torch::tensor({return_to_deck_type_data.return_to_deck_type,
                        return_to_deck_type_data.from_position},
                       make_options(device, dtype));
}

torch::Tensor vectorize_player_target_data(
    const PlayerTargetData &player_target_data,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return torch::tensor({player_target_data.player_target},
                       make_options(device, dtype));
}

torch::Tensor vectorize_payload(const PayloadVariant &payload, int64_t data_type,
                                std::optional<torch::Device> device,
                                std::optional<torch::Dtype> dtype) {
  switch (data_type) {
  case 0:
    return vectorize_attack_data(std::get<AttackData>(payload), device, dtype);
  case 1:
    return vectorize_discard_data(std::get<DiscardData>(payload), device, dtype);
  case 2:
    return vectorize_amount_data(std::get<AmountData>(payload), device, dtype);
  case 3:
    return vectorize_return_to_deck_type_data(
        std::get<ReturnToDeckTypeData>(payload), device, dtype);
  case 5:
    return vectorize_player_target_data(std::get<PlayerTargetData>(payload), device,
                                        dtype);
  default:
    throw std::invalid_argument("Unknown data type: " + std::to_string(data_type));
  }
}

FlattenInstructionsResult flatten_instructions(
    const std::string &type_key,
    const std::vector<std::vector<Instruction>> &instructions,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  std::vector<int64_t> instruction_types;
  std::vector<std::vector<int64_t>> instruction_indices;
  std::vector<int64_t> instruction_data_types;
  std::vector<std::vector<int64_t>> instruction_data_type_indices;

  FlattenInstructionsResult result;

  for (size_t batch_index = 0; batch_index < instructions.size(); ++batch_index) {
    const auto &batch_instructions = instructions[batch_index];
    for (size_t instruction_index = 0; instruction_index < batch_instructions.size();
         ++instruction_index) {
      const auto &instruction = batch_instructions[instruction_index];
      if (type_key == "InstructionType") {
        instruction_types.push_back(instruction.instruction_type);
      } else if (type_key == "ConditionType") {
        instruction_types.push_back(instruction.condition_type);
      } else {
        throw std::invalid_argument("Unknown type_key: " + type_key);
      }
      instruction_indices.push_back(
          {static_cast<int64_t>(batch_index), static_cast<int64_t>(instruction_index)});

      for (size_t data_index = 0; data_index < instruction.data.size(); ++data_index) {
        const auto &data = instruction.data[data_index];
        const auto data_type = data.instruction_data_type;
        instruction_data_types.push_back(data_type);
        instruction_data_type_indices.push_back(
            {static_cast<int64_t>(batch_index), static_cast<int64_t>(instruction_index),
             static_cast<int64_t>(data_index)});

        if (data_type == 4) {
          if (!data.filter_payload.has_value()) {
            throw std::invalid_argument(
                "InstructionDataType 4 requires filter_payload");
          }
          result.filter_data.push_back(*data.filter_payload);
        } else {
          if (!data.payload.has_value()) {
            throw std::invalid_argument(
                "InstructionDataType requires vectorizable payload");
          }
          result.instruction_data[data_type].push_back(
              vectorize_payload(*data.payload, data_type, device, dtype));
        }

        result.instruction_data_indices[data_type].push_back(
            {static_cast<int64_t>(batch_index), static_cast<int64_t>(instruction_index),
             static_cast<int64_t>(data_index)});
      }
    }
  }

  const auto options = make_options(device, dtype);
  result.instruction_types = torch::tensor(instruction_types, options);
  result.instruction_indices = tensor_utils::tensor_from_2d_int64(
      instruction_indices, device, dtype);
  result.instruction_data_types = torch::tensor(instruction_data_types, options);
  result.instruction_data_type_indices = tensor_utils::tensor_from_2d_int64(
      instruction_data_type_indices, device, dtype);

  return result;
}

} // namespace nesting
