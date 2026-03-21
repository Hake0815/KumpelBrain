#include "../include/Nesting.h"
#include "../include/TensorUtils.h"

#include <sstream>
#include <stdexcept>

namespace nesting {

namespace serialization = gamecore::serialization;

namespace {

constexpr int64_t filterInstructionDataType = 4;

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
  auto it = operators.find(group_index);
  if (it == operators.end()) {
    return std::nullopt;
  }
  return it->second;
}

const serialization::ProtoBufFilterCondition &
require_filter_condition(const serialization::ProtoBufFilter &node) {
  if (!node.has_condition()) {
    throw std::invalid_argument("Leaf filter node is missing condition");
  }
  return node.condition();
}

std::vector<int64_t> make_condition_triplet(
    const serialization::ProtoBufFilterCondition &condition) {
  return {static_cast<int64_t>(condition.field()),
          static_cast<int64_t>(condition.operation()),
          static_cast<int64_t>(condition.value())};
}

void append_filter_leaf_entry(const serialization::ProtoBufFilter &node,
                              const GroupIndex &path, int64_t logical_operator,
                              std::vector<TraverseEntry> &out) {
  out.push_back(
      TraverseEntry{make_condition_triplet(require_filter_condition(node)),
                    path, logical_operator});
}

void append_nested_filter_entries(const serialization::ProtoBufFilter &node,
                                  GroupIndex &path,
                                  std::vector<TraverseEntry> &out) {
  const auto logical_operator = static_cast<int64_t>(node.logical_operator());
  const auto &operands = node.operands();
  for (int child_index = 0; child_index < operands.size(); ++child_index) {
    const auto &child = operands.Get(child_index);
    if (child.is_leaf()) {
      append_filter_leaf_entry(child, path, logical_operator, out);
      continue;
    }

    path.push_back(static_cast<int64_t>(child_index));
    append_nested_filter_entries(child, path, out);
    path.pop_back();
  }
}

void append_root_filter_entries(
    const std::vector<serialization::ProtoBufFilter> &nested_input,
    std::vector<TraverseEntry> &out) {
  GroupIndex path;
  for (size_t root_index = 0; root_index < nested_input.size(); ++root_index) {
    const auto &node = nested_input[root_index];
    path.push_back(static_cast<int64_t>(root_index));
    if (node.is_leaf()) {
      append_filter_leaf_entry(node, path, 0, out);
    } else {
      append_nested_filter_entries(node, path, out);
    }
    path.pop_back();
  }
}

void break_down_stack(
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const GroupIndex &group_index,
    const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)>
        &combine_function);

void push_new_group(
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const GroupIndex &group_index,
    const torch::Tensor &element) {
  current_combination.push_back({element});
  current_groups.push_back(group_index);
}

void add_to_stack(
    const torch::Tensor &element,
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const GroupIndex &group_index,
    const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)>
        &combine_function) {
  if (!current_groups.empty() && group_index == current_groups.back()) {
    current_combination.back().push_back(element);
    return;
  }

  break_down_stack(current_combination, current_groups, group_index, operators,
                   combine_function);

  if (!current_groups.empty() && group_index == current_groups.back()) {
    current_combination.back().push_back(element);
    return;
  }

  push_new_group(current_combination, current_groups, group_index, element);
}

GroupIndex parent_group_index(const GroupIndex &group_index) {
  GroupIndex parent = group_index;
  if (!parent.empty()) {
    parent.pop_back();
  }
  return parent;
}

torch::Tensor combine_top_group(
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)>
        &combine_function) {
  const auto current_group = current_groups.back();
  current_groups.pop_back();
  const auto combined = combine_function(
      current_combination.back(), get_operator(operators, current_group));
  current_combination.pop_back();
  return combined;
}

void break_down_stack(
    std::vector<std::vector<torch::Tensor>> &current_combination,
    std::vector<GroupIndex> &current_groups, const GroupIndex &group_index,
    const OperatorMap &operators,
    const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                      std::optional<int64_t>)>
        &combine_function) {
  while (!current_groups.empty() &&
         !is_prefix(current_groups.back(), group_index)) {
    const auto current_group = current_groups.back();
    const auto combined = combine_top_group(current_combination, current_groups,
                                            operators, combine_function);
    add_to_stack(combined, current_combination, current_groups,
                 parent_group_index(current_group), operators,
                 combine_function);
  }
}

std::optional<torch::Dtype>
resolve_payload_dtype(std::optional<torch::Dtype> dtype) {
  if (dtype.has_value()) {
    return dtype;
  }
  return torch::kInt64;
}

std::vector<int64_t> make_instruction_index(size_t batch_index,
                                            size_t instruction_index) {
  return {static_cast<int64_t>(batch_index),
          static_cast<int64_t>(instruction_index)};
}

std::vector<int64_t> make_instruction_data_index(size_t batch_index,
                                                 size_t instruction_index,
                                                 int data_index) {
  return {static_cast<int64_t>(batch_index),
          static_cast<int64_t>(instruction_index),
          static_cast<int64_t>(data_index)};
}

std::tuple<int64_t, int64_t, int64_t>
make_instruction_data_index_tuple(size_t batch_index, size_t instruction_index,
                                  int data_index) {
  return {static_cast<int64_t>(batch_index),
          static_cast<int64_t>(instruction_index),
          static_cast<int64_t>(data_index)};
}

void append_filter_payload(const serialization::ProtoBufInstructionData &data,
                           FlattenInstructionsResult &result) {
  if (!data.has_filter_data() || !data.filter_data().has_filter()) {
    throw std::invalid_argument(
        "InstructionDataType 4 requires filter_data.filter payload");
  }
  result.filter_data.push_back({data.filter_data().filter()});
}

void append_vectorized_payload(
    const serialization::ProtoBufInstructionData &data, int64_t data_type,
    FlattenInstructionsResult &result, std::optional<torch::Device> device,
    std::optional<torch::Dtype> payload_dtype) {
  result.instruction_data[data_type].push_back(
      vectorize_payload(data, device, payload_dtype));
}

void append_instruction_data_entry(
    const serialization::ProtoBufInstructionData &data, size_t batch_index,
    size_t instruction_index, int data_index,
    std::vector<int64_t> &instruction_data_types,
    std::vector<std::vector<int64_t>> &instruction_data_type_indices,
    FlattenInstructionsResult &result, std::optional<torch::Device> device,
    std::optional<torch::Dtype> payload_dtype) {
  const auto data_type = static_cast<int64_t>(data.instruction_data_type());
  instruction_data_types.push_back(data_type);
  instruction_data_type_indices.push_back(
      make_instruction_data_index(batch_index, instruction_index, data_index));

  if (data_type == filterInstructionDataType) {
    append_filter_payload(data, result);
  } else {
    append_vectorized_payload(data, data_type, result, device, payload_dtype);
  }

  result.instruction_data_indices[data_type].push_back(
      make_instruction_data_index_tuple(batch_index, instruction_index,
                                        data_index));
}

template <typename MessageType, typename TypeAccessor>
void append_message_entries(
    const MessageType &message, size_t batch_index, size_t instruction_index,
    const TypeAccessor &type_accessor, std::vector<int64_t> &instruction_types,
    std::vector<std::vector<int64_t>> &instruction_indices,
    std::vector<int64_t> &instruction_data_types,
    std::vector<std::vector<int64_t>> &instruction_data_type_indices,
    FlattenInstructionsResult &result, std::optional<torch::Device> device,
    std::optional<torch::Dtype> payload_dtype) {
  instruction_types.push_back(type_accessor(message));
  instruction_indices.push_back(
      make_instruction_index(batch_index, instruction_index));

  for (int data_index = 0; data_index < message.data_size(); ++data_index) {
    append_instruction_data_entry(
        message.data(data_index), batch_index, instruction_index, data_index,
        instruction_data_types, instruction_data_type_indices, result, device,
        payload_dtype);
  }
}

FlattenInstructionsResult build_flatten_result(
    std::vector<int64_t> instruction_types,
    std::vector<std::vector<int64_t>> instruction_indices,
    std::vector<int64_t> instruction_data_types,
    std::vector<std::vector<int64_t>> instruction_data_type_indices,
    FlattenInstructionsResult result, std::optional<torch::Device> device,
    std::optional<torch::Dtype> dtype) {
  const auto options = make_options(device, dtype);
  result.instruction_types = torch::tensor(instruction_types, options);
  result.instruction_indices =
      tensor_utils::tensor_from_2d_int64(instruction_indices, device, dtype);
  result.instruction_data_types =
      torch::tensor(instruction_data_types, options);
  result.instruction_data_type_indices = tensor_utils::tensor_from_2d_int64(
      instruction_data_type_indices, device, dtype);
  return result;
}

template <typename MessageType, typename TypeAccessor>
FlattenInstructionsResult
flatten_messages(const std::vector<std::vector<MessageType>> &instruction_likes,
                 const TypeAccessor &type_accessor,
                 std::optional<torch::Device> device,
                 std::optional<torch::Dtype> dtype) {
  std::vector<int64_t> instruction_types;
  std::vector<std::vector<int64_t>> instruction_indices;
  std::vector<int64_t> instruction_data_types;
  std::vector<std::vector<int64_t>> instruction_data_type_indices;
  FlattenInstructionsResult result;

  const auto payload_dtype = resolve_payload_dtype(dtype);

  for (size_t batch_index = 0; batch_index < instruction_likes.size();
       ++batch_index) {
    const auto &batch_instruction_likes = instruction_likes[batch_index];
    for (size_t instruction_index = 0;
         instruction_index < batch_instruction_likes.size();
         ++instruction_index) {
      append_message_entries(
          batch_instruction_likes[instruction_index], batch_index,
          instruction_index, type_accessor, instruction_types,
          instruction_indices, instruction_data_types,
          instruction_data_type_indices, result, device, payload_dtype);
    }
  }

  return build_flatten_result(std::move(instruction_types),
                              std::move(instruction_indices),
                              std::move(instruction_data_types),
                              std::move(instruction_data_type_indices),
                              std::move(result), device, dtype);
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

std::vector<TraverseEntry> traverse_filter(
    const std::vector<serialization::ProtoBufFilter> &nested_input) {
  std::vector<TraverseEntry> out;
  append_root_filter_entries(nested_input, out);
  return out;
}

FlattenResult flatten(const std::vector<TraverseEntry> &entries) {
  FlattenResult result;
  result.flattened_input.reserve(entries.size());
  result.groups.reserve(entries.size());

  for (const auto &entry : entries) {
    result.flattened_input.push_back(entry.value);
    result.groups.push_back(entry.group_index);
    result.operators[entry.group_index] = entry.op;
  }
  return result;
}

std::vector<torch::Tensor>
reduce(const std::vector<torch::Tensor> &flattened_input,
       const std::vector<GroupIndex> &groups, const OperatorMap &operators,
       const std::function<torch::Tensor(const std::vector<torch::Tensor> &,
                                         std::optional<int64_t>)>
           &combine_function) {
  std::vector<std::vector<torch::Tensor>> current_combination;
  std::vector<GroupIndex> current_groups;
  int64_t current_batch_index = 0;

  auto start_next_batch_group = [&]() {
    current_groups.push_back(GroupIndex{current_batch_index});
    current_batch_index += 1;
    current_combination.push_back({});
  };

  for (size_t i = 0; i < groups.size(); ++i) {
    if (current_groups.empty()) {
      start_next_batch_group();
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

torch::Tensor vectorize_amount_data(
    const serialization::ProtoBufCardAmountInstructionData &amount_data,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  if (!amount_data.has_amount()) {
    throw std::invalid_argument(
        "CardAmountInstructionData requires amount field");
  }

  return torch::tensor({static_cast<int64_t>(amount_data.amount().min()),
                        static_cast<int64_t>(amount_data.amount().max()),
                        static_cast<int64_t>(amount_data.from_position())},
                       make_options(device, dtype));
}

torch::Tensor vectorize_attack_data(
    const serialization::ProtoBufAttackInstructionData &attack_data,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return torch::tensor({static_cast<int64_t>(attack_data.attack_target()),
                        static_cast<int64_t>(attack_data.damage())},
                       make_options(device, dtype));
}

torch::Tensor vectorize_discard_data(
    const serialization::ProtoBufDiscardInstructionData &discard_data,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return torch::tensor({static_cast<int64_t>(discard_data.target_source())},
                       make_options(device, dtype));
}

torch::Tensor vectorize_return_to_deck_type_data(
    const serialization::ProtoBufReturnToDeckTypeInstructionData
        &return_to_deck_type_data,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return torch::tensor(
      {static_cast<int64_t>(return_to_deck_type_data.return_to_deck_type()),
       static_cast<int64_t>(return_to_deck_type_data.from_position())},
      make_options(device, dtype));
}

torch::Tensor vectorize_player_target_data(
    const serialization::ProtoBufPlayerTargetInstructionData
        &player_target_data,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return torch::tensor(
      {static_cast<int64_t>(player_target_data.player_target())},
      make_options(device, dtype));
}

torch::Tensor
vectorize_payload(const serialization::ProtoBufInstructionData &data,
                  std::optional<torch::Device> device,
                  std::optional<torch::Dtype> dtype) {
  switch (static_cast<int64_t>(data.instruction_data_type())) {
  case 0:
    if (!data.has_attack_data()) {
      throw std::invalid_argument("InstructionDataType 0 requires attack_data");
    }
    return vectorize_attack_data(data.attack_data(), device, dtype);
  case 1:
    if (!data.has_discard_data()) {
      throw std::invalid_argument(
          "InstructionDataType 1 requires discard_data");
    }
    return vectorize_discard_data(data.discard_data(), device, dtype);
  case 2:
    if (!data.has_card_amount_data()) {
      throw std::invalid_argument(
          "InstructionDataType 2 requires card_amount_data");
    }
    return vectorize_amount_data(data.card_amount_data(), device, dtype);
  case 3:
    if (!data.has_return_to_deck_type_data()) {
      throw std::invalid_argument(
          "InstructionDataType 3 requires return_to_deck_type_data");
    }
    return vectorize_return_to_deck_type_data(data.return_to_deck_type_data(),
                                              device, dtype);
  case 5:
    if (!data.has_player_target_data()) {
      throw std::invalid_argument(
          "InstructionDataType 5 requires player_target_data");
    }
    return vectorize_player_target_data(data.player_target_data(), device,
                                        dtype);
  default:
    throw std::invalid_argument(
        "Unknown data type: " +
        std::to_string(static_cast<int64_t>(data.instruction_data_type())));
  }
}

FlattenInstructionsResult flatten_instructions(
    const std::vector<std::vector<serialization::ProtoBufInstruction>>
        &instructions,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return flatten_messages(
      instructions,
      [](const serialization::ProtoBufInstruction &instruction) {
        return static_cast<int64_t>(instruction.instruction_type());
      },
      device, dtype);
}

FlattenInstructionsResult flatten_conditions(
    const std::vector<std::vector<serialization::ProtoBufCondition>>
        &conditions,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  return flatten_messages(
      conditions,
      [](const serialization::ProtoBufCondition &condition) {
        return static_cast<int64_t>(condition.condition_type());
      },
      device, dtype);
}

} // namespace nesting
