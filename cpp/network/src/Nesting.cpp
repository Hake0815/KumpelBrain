#include "../include/Nesting.h"
#include "../include/TensorUtils.h"

#include <stdexcept>

namespace nesting {

namespace serialization = gamecore::serialization;

namespace {

constexpr int64_t kFilterDataType = 4;

struct TensorBuildOptions {
  std::optional<torch::Device> device{};
  std::optional<torch::Dtype> dtype{};
};

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

struct FilterEntryCollector {
  explicit FilterEntryCollector(std::vector<TraverseEntry> &entries)
      : entries(entries) {}

  void append_leaf(const serialization::ProtoBufFilter &node,
                   int64_t logical_operator) {
    entries.push_back(
        TraverseEntry{make_condition_triplet(require_filter_condition(node)),
                      path, logical_operator});
  }

  void append_nested(const serialization::ProtoBufFilter &node) {
    const auto logical_operator = static_cast<int64_t>(node.logical_operator());
    const auto &operands = node.operands();
    for (int child_index = 0; child_index < operands.size(); ++child_index) {
      const auto &child = operands.Get(child_index);
      if (child.is_leaf()) {
        append_leaf(child, logical_operator);
        continue;
      }

      path.push_back(static_cast<int64_t>(child_index));
      append_nested(child);
      path.pop_back();
    }
  }

  void collect_roots(
      const std::vector<serialization::ProtoBufFilter> &nested_input) {
    for (size_t root_index = 0; root_index < nested_input.size();
         ++root_index) {
      const auto &node = nested_input[root_index];
      path.push_back(static_cast<int64_t>(root_index));
      if (node.is_leaf()) {
        append_leaf(node, 0);
      } else {
        append_nested(node);
      }
      path.pop_back();
    }
  }

  std::vector<TraverseEntry> &entries;
  GroupIndex path{};
};

GroupIndex parent_group_index(const GroupIndex &group_index) {
  GroupIndex parent = group_index;
  if (!parent.empty()) {
    parent.pop_back();
  }
  return parent;
}

struct ReduceStack {
  explicit ReduceStack(const ReduceRequest &request)
      : operators(request.operators),
        combine_function(request.combine_function) {}

  void start_next_batch_group() {
    current_groups.push_back(GroupIndex{current_batch_index});
    current_batch_index += 1;
    current_combination.push_back({});
  }

  void push_new_group(const GroupIndex &group_index,
                      const torch::Tensor &element) {
    current_combination.push_back({element});
    current_groups.push_back(group_index);
  }

  torch::Tensor combine_top_group() {
    const auto current_group = current_groups.back();
    current_groups.pop_back();
    const auto combined = combine_function(
        current_combination.back(), get_operator(operators, current_group));
    current_combination.pop_back();
    return combined;
  }

  void break_down_to(const GroupIndex &group_index) {
    while (!current_groups.empty() &&
           !is_prefix(current_groups.back(), group_index)) {
      const auto current_group = current_groups.back();
      const auto combined = combine_top_group();
      add_element(combined, parent_group_index(current_group));
    }
  }

  void add_element(const torch::Tensor &element,
                   const GroupIndex &group_index) {
    if (!current_groups.empty() && group_index == current_groups.back()) {
      current_combination.back().push_back(element);
      return;
    }

    break_down_to(group_index);

    if (!current_groups.empty() && group_index == current_groups.back()) {
      current_combination.back().push_back(element);
      return;
    }

    push_new_group(group_index, element);
  }

  std::vector<std::vector<torch::Tensor>> current_combination{};
  std::vector<GroupIndex> current_groups{};
  int64_t current_batch_index = 0;
  const OperatorMap &operators;
  const ReduceCombineFunction &combine_function;
};

std::optional<torch::Dtype>
resolve_payload_dtype(std::optional<torch::Dtype> dtype) {
  if (dtype.has_value()) {
    return dtype;
  }
  return torch::kInt64;
}

constexpr std::array<int64_t, kNumInstructionDataTypes> kPayloadWidths = {
    2, 1, 3, 2, 0, 1};

int64_t payload_width(int64_t data_type) {
  if (data_type < 0 ||
      data_type >= static_cast<int64_t>(kPayloadWidths.size())) {
    throw std::invalid_argument("Unknown payload width for data type " +
                                std::to_string(data_type));
  }
  return kPayloadWidths[static_cast<size_t>(data_type)];
}

struct PayloadTensorBuilder {
  std::array<std::vector<int64_t>, kNumInstructionDataTypes> values{};
  std::array<std::vector<int64_t>, kNumInstructionDataTypes> global_positions{};
  std::array<int64_t, kNumInstructionDataTypes> row_counts{};
};

struct MessageLocation {
  size_t batch_index = 0;
  size_t instruction_index = 0;
};

struct DataLocation {
  MessageLocation message{};
  int data_index = 0;
};

int64_t
global_data_row_index(const std::vector<int64_t> &instruction_data_types) {
  return static_cast<int64_t>(instruction_data_types.size() - 1);
}

void record_payload_position(PayloadTensorBuilder &payload_builder,
                             int64_t data_type, int64_t global_row_index) {
  payload_builder.global_positions[static_cast<size_t>(data_type)].push_back(
      global_row_index);
  payload_builder.row_counts[static_cast<size_t>(data_type)] += 1;
}

void append_payload_values(PayloadTensorBuilder &payload_builder,
                           int64_t data_type,
                           const std::vector<int64_t> &payload_values) {
  const auto width = payload_width(data_type);
  if (static_cast<int64_t>(payload_values.size()) != width) {
    throw std::invalid_argument("Payload width mismatch for instruction data");
  }

  auto &buffer = payload_builder.values[static_cast<size_t>(data_type)];
  buffer.insert(buffer.end(), payload_values.begin(), payload_values.end());
}

std::vector<int64_t>
vectorize_payload_values(const serialization::ProtoBufInstructionData &data) {
  const auto data_type = static_cast<int64_t>(data.instruction_data_type());
  switch (data_type) {
  case 0:
    if (!data.has_attack_data()) {
      throw std::invalid_argument("InstructionDataType 0 requires attack_data");
    }
    return {static_cast<int64_t>(data.attack_data().attack_target()),
            static_cast<int64_t>(data.attack_data().damage())};
  case 1:
    if (!data.has_discard_data()) {
      throw std::invalid_argument(
          "InstructionDataType 1 requires discard_data");
    }
    return {static_cast<int64_t>(data.discard_data().target_source())};
  case 2:
    if (!data.has_card_amount_data() || !data.card_amount_data().has_amount()) {
      throw std::invalid_argument(
          "InstructionDataType 2 requires card_amount_data.amount");
    }
    return {static_cast<int64_t>(data.card_amount_data().amount().min()),
            static_cast<int64_t>(data.card_amount_data().amount().max()),
            static_cast<int64_t>(data.card_amount_data().from_position())};
  case 3:
    if (!data.has_return_to_deck_type_data()) {
      throw std::invalid_argument(
          "InstructionDataType 3 requires return_to_deck_type_data");
    }
    return {
        static_cast<int64_t>(
            data.return_to_deck_type_data().return_to_deck_type()),
        static_cast<int64_t>(data.return_to_deck_type_data().from_position())};
  case 5:
    if (!data.has_player_target_data()) {
      throw std::invalid_argument(
          "InstructionDataType 5 requires player_target_data");
    }
    return {static_cast<int64_t>(data.player_target_data().player_target())};
  default:
    throw std::invalid_argument(
        "Unknown data type: " +
        std::to_string(static_cast<int64_t>(data.instruction_data_type())));
  }
}

struct DensePayloadTensorSpec {
  const std::vector<int64_t> &buffer;
  int64_t rows = 0;
  int64_t width = 0;
};

torch::Tensor build_dense_payload_tensor(const DensePayloadTensorSpec &spec,
                                         const TensorBuildOptions &options) {
  auto tensor_options =
      make_options(options.device, resolve_payload_dtype(options.dtype));
  if (spec.width == 0) {
    return torch::empty({0, 0}, tensor_options);
  }
  if (spec.rows == 0) {
    return torch::empty({0, spec.width}, tensor_options);
  }
  return torch::tensor(spec.buffer, tensor_options)
      .view({spec.rows, spec.width});
}

torch::Tensor
build_dense_payload_reorder(const PayloadTensorBuilder &payload_builder,
                            const TensorBuildOptions &options) {
  int64_t total_rows = 0;
  for (const auto row_count : payload_builder.row_counts) {
    total_rows += row_count;
  }
  std::vector<int64_t> reorder(static_cast<size_t>(total_rows), 0);
  int64_t type_base = 0;
  for (size_t type_index = 0; type_index < payload_builder.row_counts.size();
       ++type_index) {
    const auto &positions = payload_builder.global_positions[type_index];
    for (size_t local_index = 0; local_index < positions.size();
         ++local_index) {
      reorder[static_cast<size_t>(positions[local_index])] =
          type_base + static_cast<int64_t>(local_index);
    }
    type_base += payload_builder.row_counts[type_index];
  }

  return torch::tensor(reorder, make_options(options.device, torch::kLong));
}

void finalize_dense_payload_tensors(FlattenInstructionsResult &result,
                                    const PayloadTensorBuilder &payload_builder,
                                    const TensorBuildOptions &options) {
  for (size_t type_index = 0;
       type_index < result.instruction_data_tensors.size(); ++type_index) {
    result.instruction_data_tensors[type_index] = build_dense_payload_tensor(
        DensePayloadTensorSpec{payload_builder.values[type_index],
                               payload_builder.row_counts[type_index],
                               kPayloadWidths[type_index]},
        options);
  }

  result.instruction_data_reorder =
      build_dense_payload_reorder(payload_builder, options);
}

struct FilterCompileBuilder {
  std::vector<bool> node_is_leaf{};
  std::vector<int64_t> node_logical_operator{};
  std::vector<int64_t> node_depth{};
  std::vector<int64_t> child_ptr{};
  std::vector<int64_t> child_idx{};
  std::vector<int64_t> leaf_node_index{};
  std::vector<int64_t> leaf_field{};
  std::vector<int64_t> leaf_compare_op{};
  std::vector<int64_t> leaf_value{};
  std::vector<int64_t> root_node_index{};
};

void append_leaf_condition(const serialization::ProtoBufFilter &node,
                           int64_t node_id, FilterCompileBuilder &builder) {
  const auto &condition = require_filter_condition(node);
  builder.leaf_node_index.push_back(node_id);
  builder.leaf_field.push_back(static_cast<int64_t>(condition.field()));
  builder.leaf_compare_op.push_back(
      static_cast<int64_t>(condition.operation()));
  builder.leaf_value.push_back(static_cast<int64_t>(condition.value()));
}

int64_t append_compiled_filter_node(const serialization::ProtoBufFilter &node,
                                    int64_t depth,
                                    FilterCompileBuilder &builder) {
  std::vector<int64_t> child_nodes;
  child_nodes.reserve(node.operands_size());
  for (int child_index = 0; child_index < node.operands_size(); ++child_index) {
    child_nodes.push_back(append_compiled_filter_node(
        node.operands(child_index), depth + 1, builder));
  }

  const auto node_id = static_cast<int64_t>(builder.node_is_leaf.size());
  const auto child_start = static_cast<int64_t>(builder.child_idx.size());
  builder.child_idx.insert(builder.child_idx.end(), child_nodes.begin(),
                           child_nodes.end());
  builder.child_ptr.push_back(child_start);
  builder.node_is_leaf.push_back(node.is_leaf());
  builder.node_logical_operator.push_back(
      node.is_leaf() ? 0 : static_cast<int64_t>(node.logical_operator()));
  builder.node_depth.push_back(depth);

  if (node.is_leaf()) {
    append_leaf_condition(node, node_id, builder);
  }

  return node_id;
}

torch::Tensor tensor_from_bool_vector(const std::vector<bool> &values,
                                      std::optional<torch::Device> device) {
  std::vector<uint8_t> dense(values.begin(), values.end());
  return torch::tensor(dense, torch::TensorOptions().dtype(torch::kBool))
      .to(device.value_or(torch::Device(torch::kCPU)));
}

FilterBatchTensors
make_filter_batch_tensors(const FilterCompileBuilder &builder,
                          const TensorBuildOptions &options) {
  const auto index_dtype = resolve_payload_dtype(options.dtype);
  FilterBatchTensors out;
  out.node_is_leaf =
      tensor_from_bool_vector(builder.node_is_leaf, options.device);
  out.node_logical_operator = torch::tensor(
      builder.node_logical_operator, make_options(options.device, index_dtype));
  out.node_depth = torch::tensor(builder.node_depth,
                                 make_options(options.device, index_dtype));
  out.child_ptr = torch::tensor(builder.child_ptr,
                                make_options(options.device, index_dtype));
  out.child_idx = torch::tensor(builder.child_idx,
                                make_options(options.device, index_dtype));
  out.leaf_node_index = torch::tensor(
      builder.leaf_node_index, make_options(options.device, index_dtype));
  out.leaf_field = torch::tensor(builder.leaf_field,
                                 make_options(options.device, index_dtype));
  out.leaf_compare_op = torch::tensor(
      builder.leaf_compare_op, make_options(options.device, index_dtype));
  out.leaf_value = torch::tensor(builder.leaf_value,
                                 make_options(options.device, index_dtype));
  out.root_node_index = torch::tensor(
      builder.root_node_index, make_options(options.device, index_dtype));
  return out;
}

std::vector<int64_t> make_instruction_index(const MessageLocation &location) {
  return {static_cast<int64_t>(location.batch_index),
          static_cast<int64_t>(location.instruction_index)};
}

std::vector<int64_t> make_instruction_data_index(const DataLocation &location) {
  return {static_cast<int64_t>(location.message.batch_index),
          static_cast<int64_t>(location.message.instruction_index),
          static_cast<int64_t>(location.data_index)};
}

void append_vectorized_payload(
    const serialization::ProtoBufInstructionData &data, int64_t data_type,
    PayloadTensorBuilder &payload_builder) {
  append_payload_values(payload_builder, data_type,
                        vectorize_payload_values(data));
}

template <typename MessageType, typename TypeAccessor>
struct FlatMessageBuilder {
  explicit FlatMessageBuilder(const TypeAccessor &type_accessor)
      : type_accessor(type_accessor) {}

  void
  append_instruction_data(const serialization::ProtoBufInstructionData &data,
                          const DataLocation &location, int64_t parent_row) {
    const auto data_type = static_cast<int64_t>(data.instruction_data_type());
    instruction_data_types.push_back(data_type);
    instruction_data_parent_rows.push_back(parent_row);
    instruction_data_type_indices.push_back(
        make_instruction_data_index(location));

    if (data_type == kFilterDataType) {
      if (!data.has_filter_data() || !data.filter_data().has_filter()) {
        throw std::invalid_argument(
            "InstructionDataType 4 requires filter_data.filter payload");
      }
      filter_data.push_back({data.filter_data().filter()});
    } else {
      append_vectorized_payload(data, data_type, payload_builder);
    }

    // Filters (type 4) have width 0 in the payload tensor but still occupy
    // positions in the reorder mapping so their embeddings are interleaved
    // with other data-type embeddings in the original order.
    record_payload_position(payload_builder, data_type,
                            global_data_row_index(instruction_data_types));
  }

  void append_message(const MessageType &message,
                      const MessageLocation &location) {
    instruction_types.push_back(type_accessor(message));
    instruction_indices.push_back(make_instruction_index(location));
    const auto parent_row = static_cast<int64_t>(instruction_types.size() - 1);

    for (int data_index = 0; data_index < message.data_size(); ++data_index) {
      append_instruction_data(message.data(data_index),
                              DataLocation{location, data_index}, parent_row);
    }
  }

  FlattenInstructionsResult build(const TensorBuildOptions &options) {
    FlattenInstructionsResult result;
    const auto tensor_options = make_options(options.device, options.dtype);
    result.instruction_types = torch::tensor(instruction_types, tensor_options);
    result.instruction_indices = tensor_utils::tensor_from_2d_int64(
        instruction_indices, options.device, options.dtype);
    result.instruction_data_types =
        torch::tensor(instruction_data_types, tensor_options);
    result.instruction_data_parent_rows =
        torch::tensor(instruction_data_parent_rows, tensor_options);
    result.instruction_data_type_indices = tensor_utils::tensor_from_2d_int64(
        instruction_data_type_indices, options.device, options.dtype);
    finalize_dense_payload_tensors(result, payload_builder, options);
    result.filter_batch =
        compile_filter_batch(filter_data, options.device, options.dtype);
    return result;
  }

  const TypeAccessor &type_accessor;
  std::vector<int64_t> instruction_types{};
  std::vector<std::vector<int64_t>> instruction_indices{};
  std::vector<int64_t> instruction_data_types{};
  std::vector<int64_t> instruction_data_parent_rows{};
  std::vector<std::vector<int64_t>> instruction_data_type_indices{};
  std::vector<std::vector<serialization::ProtoBufFilter>> filter_data{};
  PayloadTensorBuilder payload_builder{};
};

template <typename MessageType, typename TypeAccessor>
FlattenInstructionsResult
flatten_messages(const std::vector<std::vector<MessageType>> &instruction_likes,
                 const TypeAccessor &type_accessor,
                 const TensorBuildOptions &options) {
  FlatMessageBuilder<MessageType, TypeAccessor> builder(type_accessor);

  for (size_t batch_index = 0; batch_index < instruction_likes.size();
       ++batch_index) {
    const auto &batch_instruction_likes = instruction_likes[batch_index];
    for (size_t instruction_index = 0;
         instruction_index < batch_instruction_likes.size();
         ++instruction_index) {
      builder.append_message(batch_instruction_likes[instruction_index],
                             MessageLocation{batch_index, instruction_index});
    }
  }

  return builder.build(options);
}

torch::Tensor move_tensor_if_defined(const torch::Tensor &tensor,
                                     torch::Device device) {
  return tensor.defined() ? tensor.to(device) : tensor;
}

} // namespace

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
  FilterEntryCollector collector(out);
  collector.collect_roots(nested_input);
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

std::vector<torch::Tensor> reduce(const ReduceRequest &request) {
  ReduceStack stack(request);

  for (size_t i = 0; i < request.groups.size(); ++i) {
    if (stack.current_groups.empty()) {
      stack.start_next_batch_group();
    }

    stack.add_element(request.flattened_input[i], request.groups[i]);
  }

  stack.break_down_to(GroupIndex{});
  if (stack.current_combination.empty()) {
    return {};
  }
  return stack.current_combination.back();
}

FilterBatchTensors compile_filter_batch(
    const std::vector<std::vector<serialization::ProtoBufFilter>> &filters,
    std::optional<torch::Device> device, std::optional<torch::Dtype> dtype) {
  FilterCompileBuilder builder;
  for (const auto &forest : filters) {
    for (const auto &root : forest) {
      builder.root_node_index.push_back(
          append_compiled_filter_node(root, 0, builder));
    }
  }
  builder.child_ptr.push_back(static_cast<int64_t>(builder.child_idx.size()));
  return make_filter_batch_tensors(builder, TensorBuildOptions{device, dtype});
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
      TensorBuildOptions{device, dtype});
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
      TensorBuildOptions{device, dtype});
}

FlattenInstructionsResult
move_flattened_result_to_device(const FlattenInstructionsResult &result,
                                torch::Device device) {
  FlattenInstructionsResult moved;
  moved.instruction_types = result.instruction_types.to(device);
  moved.instruction_indices = result.instruction_indices.to(device);
  moved.instruction_data_types = result.instruction_data_types.to(device);
  moved.instruction_data_parent_rows =
      result.instruction_data_parent_rows.to(device);
  moved.instruction_data_type_indices =
      result.instruction_data_type_indices.to(device);
  moved.instruction_data_reorder = result.instruction_data_reorder.to(device);
  moved.filter_batch = move_filter_batch_to_device(result.filter_batch, device);

  for (size_t i = 0; i < moved.instruction_data_tensors.size(); ++i) {
    moved.instruction_data_tensors[i] =
        result.instruction_data_tensors[i].to(device);
  }

  return moved;
}

FilterBatchTensors move_filter_batch_to_device(const FilterBatchTensors &batch,
                                               torch::Device device) {
  FilterBatchTensors moved;
  moved.node_is_leaf = move_tensor_if_defined(batch.node_is_leaf, device);
  moved.node_logical_operator =
      move_tensor_if_defined(batch.node_logical_operator, device);
  moved.node_depth = move_tensor_if_defined(batch.node_depth, device);
  moved.child_ptr = move_tensor_if_defined(batch.child_ptr, device);
  moved.child_idx = move_tensor_if_defined(batch.child_idx, device);
  moved.leaf_node_index = move_tensor_if_defined(batch.leaf_node_index, device);
  moved.leaf_field = move_tensor_if_defined(batch.leaf_field, device);
  moved.leaf_compare_op = move_tensor_if_defined(batch.leaf_compare_op, device);
  moved.leaf_value = move_tensor_if_defined(batch.leaf_value, device);
  moved.root_node_index = move_tensor_if_defined(batch.root_node_index, device);
  return moved;
}

} // namespace nesting
