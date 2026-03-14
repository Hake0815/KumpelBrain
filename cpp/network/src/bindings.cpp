#include "../include/AttackDataEmbedding.h"
#include "../include/CardAmountDataEmbedding.h"
#include "../include/ConditionEmbedding.h"
#include "../include/DiscardDataEmbedding.h"
#include "../include/FilterConditionEmbedding.h"
#include "../include/FilterEmbedding.h"
#include "../include/InstructionDataEmbedding.h"
#include "../include/InstructionEmbedding.h"
#include "../include/MultiHeadAttention.h"
#include "../include/Nesting.h"
#include "../include/NormalizedLinear.h"
#include "../include/PlayerTargetDataEmbedding.h"
#include "../include/PositionalEmbedding.h"
#include "../include/ReturnToDeckTypeDataEmbedding.h"
#include "../include/SharedEmbeddingHolder.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace {

nesting::GroupIndex py_tuple_to_group_index(const pybind11::tuple &t) {
  nesting::GroupIndex group_index;
  group_index.reserve(t.size());
  for (auto item : t) {
    group_index.push_back(pybind11::cast<int64_t>(item));
  }
  return group_index;
}

pybind11::tuple
group_index_to_py_tuple(const nesting::GroupIndex &group_index) {
  pybind11::tuple out(group_index.size());
  for (size_t i = 0; i < group_index.size(); ++i) {
    out[i] = pybind11::int_(group_index[i]);
  }
  return out;
}

nesting::FilterNode parse_filter_node(const pybind11::dict &node_dict) {
  nesting::FilterNode node;
  node.is_leaf = pybind11::cast<bool>(node_dict["IsLeaf"]);
  node.logical_operator = pybind11::cast<int64_t>(node_dict["LogicalOperator"]);

  if (node.is_leaf) {
    auto condition_obj = node_dict["Condition"];
    if (!condition_obj.is_none()) {
      auto condition_dict = pybind11::cast<pybind11::dict>(condition_obj);
      node.condition.field = pybind11::cast<int64_t>(condition_dict["Field"]);
      node.condition.operation =
          pybind11::cast<int64_t>(condition_dict["Operation"]);
      node.condition.value = pybind11::cast<int64_t>(condition_dict["Value"]);
    }
  } else {
    auto operands_obj = node_dict["Operands"];
    for (auto operand : pybind11::cast<pybind11::list>(operands_obj)) {
      node.operands.push_back(parse_filter_node(
          pybind11::cast<pybind11::dict>(operand.cast<pybind11::object>())));
    }
  }

  return node;
}

std::vector<nesting::FilterNode>
parse_filter_list(const pybind11::iterable &nested_input) {
  std::vector<nesting::FilterNode> nodes;
  for (auto item : nested_input) {
    nodes.push_back(parse_filter_node(
        pybind11::cast<pybind11::dict>(item.cast<pybind11::object>())));
  }
  return nodes;
}

std::array<std::vector<torch::Tensor>, 6>
parse_instruction_data_tensors(const pybind11::tuple &instruction_data) {
  if (instruction_data.size() != 6) {
    throw std::invalid_argument("instruction_data must have length 6");
  }

  std::array<std::vector<torch::Tensor>, 6> parsed;
  for (size_t i = 0; i < 6; ++i) {
    if (i == 4) {
      continue;
    }
    auto items = pybind11::cast<pybind11::list>(instruction_data[i]);
    for (auto item : items) {
      parsed[i].push_back(
          pybind11::cast<torch::Tensor>(item.cast<pybind11::object>()));
    }
  }
  return parsed;
}

std::vector<std::vector<nesting::FilterNode>>
parse_instruction_filter_data(const pybind11::tuple &instruction_data) {
  auto filters = pybind11::cast<pybind11::list>(instruction_data[4]);
  std::vector<std::vector<nesting::FilterNode>> nodes;
  nodes.reserve(filters.size());
  for (auto item : filters) {
    auto obj = item.cast<pybind11::object>();
    if (pybind11::isinstance<pybind11::dict>(obj)) {
      nodes.push_back({parse_filter_node(pybind11::cast<pybind11::dict>(obj))});
    } else if (pybind11::isinstance<pybind11::list>(obj) ||
               pybind11::isinstance<pybind11::tuple>(obj)) {
      nodes.push_back(
          parse_filter_list(pybind11::cast<pybind11::iterable>(obj)));
    } else {
      throw std::invalid_argument(
          "instruction_data[4] items must be filter dict or list");
    }
  }
  return nodes;
}

std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6>
parse_instruction_data_indices(
    const pybind11::tuple &instruction_data_indices) {
  if (instruction_data_indices.size() != 6) {
    throw std::invalid_argument("instruction_data_indices must have length 6");
  }

  std::array<std::vector<std::tuple<int64_t, int64_t, int64_t>>, 6> parsed;
  for (size_t i = 0; i < 6; ++i) {
    auto idx_list = pybind11::cast<pybind11::list>(instruction_data_indices[i]);
    parsed[i].reserve(idx_list.size());
    for (auto item : idx_list) {
      auto tup = pybind11::cast<pybind11::tuple>(item);
      if (tup.size() != 3) {
        throw std::invalid_argument("Each instruction index must be length 3");
      }
      parsed[i].emplace_back(pybind11::cast<int64_t>(tup[0]),
                             pybind11::cast<int64_t>(tup[1]),
                             pybind11::cast<int64_t>(tup[2]));
    }
  }
  return parsed;
}

nesting::PayloadVariant parse_payload_variant(int64_t data_type,
                                              const pybind11::dict &payload) {
  switch (data_type) {
  case 0:
    return nesting::AttackData{pybind11::cast<int64_t>(payload["AttackTarget"]),
                               pybind11::cast<int64_t>(payload["Damage"])};
  case 1:
    return nesting::DiscardData{
        pybind11::cast<int64_t>(payload["TargetSource"])};
  case 2: {
    auto amount = pybind11::cast<pybind11::dict>(payload["Amount"]);
    return nesting::AmountData{
        pybind11::cast<int64_t>(amount["Min"]),
        pybind11::cast<int64_t>(amount["Max"]),
        pybind11::cast<int64_t>(payload["FromPosition"])};
  }
  case 3:
    return nesting::ReturnToDeckTypeData{
        pybind11::cast<int64_t>(payload["ReturnToDeckType"]),
        pybind11::cast<int64_t>(payload["FromPosition"])};
  case 5:
    return nesting::PlayerTargetData{
        pybind11::cast<int64_t>(payload["PlayerTarget"])};
  default:
    throw std::invalid_argument("Unknown InstructionDataType payload");
  }
}

nesting::InstructionData
parse_instruction_data_entry(const pybind11::dict &entry) {
  nesting::InstructionData data;
  data.instruction_data_type =
      pybind11::cast<int64_t>(entry["InstructionDataType"]);
  auto payload = pybind11::cast<pybind11::dict>(entry["Payload"]);

  if (data.instruction_data_type == 4) {
    auto filter_obj = payload["Filter"].cast<pybind11::object>();
    if (pybind11::isinstance<pybind11::dict>(filter_obj)) {
      data.filter_payload = std::vector<nesting::FilterNode>{
          parse_filter_node(pybind11::cast<pybind11::dict>(filter_obj))};
    } else {
      data.filter_payload =
          parse_filter_list(pybind11::cast<pybind11::iterable>(filter_obj));
    }
  } else {
    data.payload = parse_payload_variant(data.instruction_data_type, payload);
  }

  return data;
}

nesting::Instruction parse_instruction(const pybind11::dict &instruction_dict) {
  nesting::Instruction instruction;
  if (instruction_dict.contains("InstructionType")) {
    instruction.instruction_type =
        pybind11::cast<int64_t>(instruction_dict["InstructionType"]);
  }
  if (instruction_dict.contains("ConditionType")) {
    instruction.condition_type =
        pybind11::cast<int64_t>(instruction_dict["ConditionType"]);
  }

  auto data_list = pybind11::cast<pybind11::list>(instruction_dict["Data"]);
  instruction.data.reserve(data_list.size());
  for (auto item : data_list) {
    instruction.data.push_back(parse_instruction_data_entry(
        pybind11::cast<pybind11::dict>(item.cast<pybind11::object>())));
  }
  return instruction;
}

std::vector<std::vector<nesting::Instruction>>
parse_instructions_batch(const pybind11::iterable &instructions_batch) {
  std::vector<std::vector<nesting::Instruction>> parsed;
  for (auto batch_item : instructions_batch) {
    auto batch_list =
        pybind11::cast<pybind11::list>(batch_item.cast<pybind11::object>());
    std::vector<nesting::Instruction> batch;
    batch.reserve(batch_list.size());
    for (auto instruction_item : batch_list) {
      batch.push_back(parse_instruction(pybind11::cast<pybind11::dict>(
          instruction_item.cast<pybind11::object>())));
    }
    parsed.push_back(std::move(batch));
  }
  return parsed;
}

} // namespace

// Expose class to Python
PYBIND11_MODULE(kumpel_embedding, m) {
  pybind11::module_::import("torch");
  pybind11::class_<PositionalEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<PositionalEmbeddingImpl>>(
      m, "PositionalEmbedding")
      .def(pybind11::init<int64_t, double, int64_t, torch::Device,
                          torch::Dtype>(),
           pybind11::arg("d_model"), pybind11::arg("dropout") = 0.1,
           pybind11::arg("max_len") = 5000,
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &PositionalEmbeddingImpl::forward);
  pybind11::class_<MultiHeadAttentionImpl, torch::nn::Module,
                   std::shared_ptr<MultiHeadAttentionImpl>>(
      m, "MultiHeadAttention")
      .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, int64_t, double,
                          bool, torch::Device, torch::Dtype>(),
           pybind11::arg("d_q"), pybind11::arg("d_k"), pybind11::arg("d_v"),
           pybind11::arg("d_head"), pybind11::arg("nheads"),
           pybind11::arg("dropout") = 0.0, pybind11::arg("bias") = true,
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &MultiHeadAttentionImpl::forward)
      .def("save_weights", &MultiHeadAttentionImpl::save_weights)
      .def("load_weights", &MultiHeadAttentionImpl::load_weights);
  pybind11::class_<NormalizedLinearImpl, torch::nn::Module,
                   std::shared_ptr<NormalizedLinearImpl>>(m, "NormalizedLinear")
      .def(pybind11::init<int64_t, int64_t, double, torch::Device,
                          torch::Dtype>(),
           pybind11::arg("d_in"), pybind11::arg("d_out"),
           pybind11::arg("divisor") = 400.0,
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &NormalizedLinearImpl::forward)
      .def("save_weights", &NormalizedLinearImpl::save_weights)
      .def("load_weights", &NormalizedLinearImpl::load_weights);
  pybind11::class_<SharedEmbeddingHolderImpl, torch::nn::Module,
                   std::shared_ptr<SharedEmbeddingHolderImpl>>(
      m, "SharedEmbeddingHolder")
      .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("save_weights", &SharedEmbeddingHolderImpl::save_weights)
      .def("load_weights", &SharedEmbeddingHolderImpl::load_weights);
  pybind11::class_<AttackDataEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<AttackDataEmbeddingImpl>>(
      m, "AttackDataEmbedding")
      .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &AttackDataEmbeddingImpl::forward)
      .def("save_weights", &AttackDataEmbeddingImpl::save_weights)
      .def("load_weights", &AttackDataEmbeddingImpl::load_weights);
  pybind11::class_<DiscardDataEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<DiscardDataEmbeddingImpl>>(
      m, "DiscardDataEmbedding")
      .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &DiscardDataEmbeddingImpl::forward)
      .def("save_weights", &DiscardDataEmbeddingImpl::save_weights)
      .def("load_weights", &DiscardDataEmbeddingImpl::load_weights);
  pybind11::class_<CardAmountDataEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<CardAmountDataEmbeddingImpl>>(
      m, "CardAmountDataEmbedding")
      .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &CardAmountDataEmbeddingImpl::forward)
      .def("save_weights", &CardAmountDataEmbeddingImpl::save_weights)
      .def("load_weights", &CardAmountDataEmbeddingImpl::load_weights);
  pybind11::class_<ReturnToDeckTypeDataEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<ReturnToDeckTypeDataEmbeddingImpl>>(
      m, "ReturnToDeckTypeDataEmbedding")
      .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &ReturnToDeckTypeDataEmbeddingImpl::forward)
      .def("save_weights", &ReturnToDeckTypeDataEmbeddingImpl::save_weights)
      .def("load_weights", &ReturnToDeckTypeDataEmbeddingImpl::load_weights);
  pybind11::class_<PlayerTargetDataEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<PlayerTargetDataEmbeddingImpl>>(
      m, "PlayerTargetDataEmbedding")
      .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &PlayerTargetDataEmbeddingImpl::forward)
      .def("save_weights", &PlayerTargetDataEmbeddingImpl::save_weights)
      .def("load_weights", &PlayerTargetDataEmbeddingImpl::load_weights);
  pybind11::class_<FilterConditionEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<FilterConditionEmbeddingImpl>>(
      m, "FilterConditionEmbedding")
      .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward", &FilterConditionEmbeddingImpl::forward)
      .def("save_weights", &FilterConditionEmbeddingImpl::save_weights)
      .def("load_weights", &FilterConditionEmbeddingImpl::load_weights);
  pybind11::class_<FilterEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<FilterEmbeddingImpl>>(m, "FilterEmbedding")
      .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward",
           [](FilterEmbeddingImpl &self, const pybind11::iterable &filter) {
             return self.forward(parse_filter_list(filter));
           })
      .def("save_weights", &FilterEmbeddingImpl::save_weights)
      .def("load_weights", &FilterEmbeddingImpl::load_weights);
  pybind11::class_<InstructionDataEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<InstructionDataEmbeddingImpl>>(
      m, "InstructionDataEmbedding")
      .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward",
           [](InstructionDataEmbeddingImpl &self,
              const torch::Tensor &instruction_indices,
              const torch::Tensor &instruction_data_types,
              const torch::Tensor &instruction_data_type_indices,
              const pybind11::tuple &instruction_data,
              const pybind11::tuple &instruction_data_indices,
              int64_t batch_size) {
             auto parsed_data =
                 parse_instruction_data_tensors(instruction_data);
             auto filter_data = parse_instruction_filter_data(instruction_data);
             auto parsed_indices =
                 parse_instruction_data_indices(instruction_data_indices);
             return self.forward(instruction_indices, instruction_data_types,
                                 instruction_data_type_indices, parsed_data,
                                 filter_data, parsed_indices, batch_size);
           })
      .def("save_weights", &InstructionDataEmbeddingImpl::save_weights)
      .def("load_weights", &InstructionDataEmbeddingImpl::load_weights);
  pybind11::class_<InstructionEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<InstructionEmbeddingImpl>>(
      m, "InstructionEmbedding")
      .def(pybind11::init<std::shared_ptr<InstructionDataEmbeddingImpl>,
                          std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("instruction_data_embedding"),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward",
           [](InstructionEmbeddingImpl &self,
              const pybind11::iterable &instructions_batch) {
             auto nesting_module = pybind11::module_::import("nesting");
             auto flattened = nesting_module.attr("flatten_instructions")(
                 "InstructionType", instructions_batch);
             auto flat_tuple = pybind11::cast<pybind11::tuple>(flattened);
             if (flat_tuple.size() != 6) {
               throw std::invalid_argument(
                   "nesting.flatten_instructions must return 6-tuple");
             }

             auto instruction_types =
                 pybind11::cast<torch::Tensor>(flat_tuple[0]);
             auto instruction_indices =
                 pybind11::cast<torch::Tensor>(flat_tuple[1]);
             auto instruction_data_types =
                 pybind11::cast<torch::Tensor>(flat_tuple[2]);
             auto instruction_data_type_indices =
                 pybind11::cast<torch::Tensor>(flat_tuple[3]);
             auto instruction_data =
                 pybind11::cast<pybind11::tuple>(flat_tuple[4]);
             auto instruction_data_indices =
                 pybind11::cast<pybind11::tuple>(flat_tuple[5]);

             auto parsed_data =
                 parse_instruction_data_tensors(instruction_data);
             auto filter_data = parse_instruction_filter_data(instruction_data);
             auto parsed_indices =
                 parse_instruction_data_indices(instruction_data_indices);

             const int64_t batch_size =
                 static_cast<int64_t>(pybind11::len(instructions_batch));

             return self.forward_flattened(
                 instruction_types, instruction_indices, instruction_data_types,
                 instruction_data_type_indices, parsed_data, filter_data,
                 parsed_indices, batch_size);
           })
      .def("compute_data_tensors",
           [](InstructionEmbeddingImpl &self,
              const pybind11::iterable &instructions_batch) {
             auto nesting_module = pybind11::module_::import("nesting");
             auto flattened = nesting_module.attr("flatten_instructions")(
                 "InstructionType", instructions_batch);
             auto flat_tuple = pybind11::cast<pybind11::tuple>(flattened);
             auto instruction_indices =
                 pybind11::cast<torch::Tensor>(flat_tuple[1]);
             auto instruction_data_types =
                 pybind11::cast<torch::Tensor>(flat_tuple[2]);
             auto instruction_data_type_indices =
                 pybind11::cast<torch::Tensor>(flat_tuple[3]);
             auto instruction_data =
                 pybind11::cast<pybind11::tuple>(flat_tuple[4]);
             auto instruction_data_indices =
                 pybind11::cast<pybind11::tuple>(flat_tuple[5]);

             auto parsed_data =
                 parse_instruction_data_tensors(instruction_data);
             auto filter_data = parse_instruction_filter_data(instruction_data);
             auto parsed_indices =
                 parse_instruction_data_indices(instruction_data_indices);
             const int64_t batch_size =
                 static_cast<int64_t>(pybind11::len(instructions_batch));
             return self.compute_data_tensors(
                 instruction_indices, instruction_data_types,
                 instruction_data_type_indices, parsed_data, filter_data,
                 parsed_indices, batch_size);
           })
      .def("compute_instruction_embeddings",
           [](InstructionEmbeddingImpl &self,
              const pybind11::iterable &instructions_batch) {
             auto nesting_module = pybind11::module_::import("nesting");
             auto flattened = nesting_module.attr("flatten_instructions")(
                 "InstructionType", instructions_batch);
             auto flat_tuple = pybind11::cast<pybind11::tuple>(flattened);
             auto instruction_types =
                 pybind11::cast<torch::Tensor>(flat_tuple[0]);
             auto instruction_indices =
                 pybind11::cast<torch::Tensor>(flat_tuple[1]);
             auto instruction_data_types =
                 pybind11::cast<torch::Tensor>(flat_tuple[2]);
             auto instruction_data_type_indices =
                 pybind11::cast<torch::Tensor>(flat_tuple[3]);
             auto instruction_data =
                 pybind11::cast<pybind11::tuple>(flat_tuple[4]);
             auto instruction_data_indices =
                 pybind11::cast<pybind11::tuple>(flat_tuple[5]);
             auto parsed_data =
                 parse_instruction_data_tensors(instruction_data);
             auto filter_data = parse_instruction_filter_data(instruction_data);
             auto parsed_indices =
                 parse_instruction_data_indices(instruction_data_indices);
             const int64_t batch_size =
                 static_cast<int64_t>(pybind11::len(instructions_batch));
             auto data_tensors = self.compute_data_tensors(
                 instruction_indices, instruction_data_types,
                 instruction_data_type_indices, parsed_data, filter_data,
                 parsed_indices, batch_size);
             return self.compute_instruction_embeddings(
                 instruction_types, instruction_indices,
                 instruction_data_type_indices, data_tensors);
           })
      .def("save_weights", &InstructionEmbeddingImpl::save_weights)
      .def("load_weights", &InstructionEmbeddingImpl::load_weights);
  pybind11::class_<ConditionEmbeddingImpl, torch::nn::Module,
                   std::shared_ptr<ConditionEmbeddingImpl>>(m,
                                                            "ConditionEmbedding")
      .def(pybind11::init<std::shared_ptr<InstructionDataEmbeddingImpl>,
                          std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t,
                          torch::Device, torch::Dtype>(),
           pybind11::arg("instruction_data_embedding"),
           pybind11::arg("shared_embedding_holder"),
           pybind11::arg("dimension_out"),
           pybind11::arg("device") = torch::Device(torch::kCPU),
           pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
      .def("forward",
           [](ConditionEmbeddingImpl &self,
              const pybind11::iterable &conditions_batch) {
             return self.forward(parse_instructions_batch(conditions_batch));
           })
      .def("save_weights", &ConditionEmbeddingImpl::save_weights)
      .def("load_weights", &ConditionEmbeddingImpl::load_weights);

  m.def("nesting_traverse_filter", [](const pybind11::iterable &nested_input) {
    auto nodes = parse_filter_list(nested_input);
    auto entries = nesting::traverse_filter(nodes);
    pybind11::list out;
    for (const auto &entry : entries) {
      out.append(pybind11::make_tuple(
          entry.value, group_index_to_py_tuple(entry.group_index), entry.op));
    }
    return out;
  });

  m.def("nesting_flatten_filter", [](const pybind11::iterable &nested_input) {
    auto nodes = parse_filter_list(nested_input);
    auto entries = nesting::traverse_filter(nodes);
    auto result = nesting::flatten(entries);

    pybind11::list groups;
    pybind11::dict operators;
    for (const auto &entry : entries) {
      auto key = group_index_to_py_tuple(entry.group_index);
      operators[key] = pybind11::int_(entry.op);
    }
    for (const auto &group : result.groups) {
      groups.append(group_index_to_py_tuple(group));
    }

    return pybind11::make_tuple(result.flattened_input, groups, operators);
  });

  m.def("nesting_is_prefix",
        [](const pybind11::tuple &prefix, const pybind11::tuple &test) {
          return nesting::is_prefix(py_tuple_to_group_index(prefix),
                                    py_tuple_to_group_index(test));
        });

  m.def("nesting_reduce", [](const torch::Tensor &flattened_input,
                             const pybind11::list &groups,
                             const pybind11::dict &operators,
                             pybind11::function combine_function) {
    std::vector<nesting::GroupIndex> cpp_groups;
    cpp_groups.reserve(groups.size());
    for (auto item : groups) {
      cpp_groups.push_back(
          py_tuple_to_group_index(pybind11::cast<pybind11::tuple>(item)));
    }

    nesting::OperatorMap cpp_operators;
    for (auto item : operators) {
      auto key_tuple = pybind11::cast<pybind11::tuple>(item.first);
      auto key = nesting::group_index_key(py_tuple_to_group_index(key_tuple));
      cpp_operators[key] = pybind11::cast<int64_t>(item.second);
    }

    std::vector<torch::Tensor> cpp_flattened;
    cpp_flattened.reserve(flattened_input.size(0));
    for (int64_t i = 0; i < flattened_input.size(0); ++i) {
      cpp_flattened.push_back(flattened_input[i]);
    }

    auto reduced = nesting::reduce(
        cpp_flattened, cpp_groups, cpp_operators,
        [&combine_function](const std::vector<torch::Tensor> &values,
                            std::optional<int64_t> op) {
          pybind11::list py_values;
          for (const auto &value : values) {
            py_values.append(value);
          }
          pybind11::object py_op =
              op.has_value() ? pybind11::cast(*op) : pybind11::none();
          return combine_function(py_values, py_op).cast<torch::Tensor>();
        });

    pybind11::list out;
    for (const auto &tensor : reduced) {
      out.append(tensor);
    }
    return out;
  });
}