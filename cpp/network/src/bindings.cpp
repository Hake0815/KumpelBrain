#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../include/AttackDataEmbedding.h"
#include "../include/CardAmountDataEmbedding.h"
#include "../include/CardEmbedding.h"
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

namespace {

namespace serialization = gamecore::serialization;

nesting::GroupIndex py_tuple_to_group_index(const pybind11::tuple& t) {
    nesting::GroupIndex group_index;
    group_index.reserve(t.size());
    for (auto item : t) {
        group_index.push_back(pybind11::cast<int64_t>(item));
    }
    return group_index;
}

pybind11::tuple group_index_to_py_tuple(const nesting::GroupIndex& group_index) {
    pybind11::tuple out(group_index.size());
    for (size_t i = 0; i < group_index.size(); ++i) {
        out[i] = pybind11::int_(group_index[i]);
    }
    return out;
}

template <typename MessageType>
MessageType parse_serialized_message(const pybind11::handle& input, const char* name) {
    std::string serialized;
    if (pybind11::isinstance<pybind11::bytes>(input) || pybind11::isinstance<pybind11::str>(input)) {
        serialized = pybind11::cast<std::string>(input);
    } else {
        throw std::invalid_argument(std::string(name) + " must be serialized protobuf bytes/string");
    }

    MessageType message;
    if (!message.ParseFromString(serialized)) {
        throw std::invalid_argument(std::string("Failed to parse serialized ") + name);
    }
    return message;
}

serialization::ProtoBufFilter parse_filter_object(const pybind11::handle& input);

serialization::ProtoBufFilterCondition parse_filter_condition_dict(const pybind11::dict& condition_dict) {
    serialization::ProtoBufFilterCondition condition;
    condition.set_field(
        static_cast<serialization::ProtoBufFilterType>(pybind11::cast<int64_t>(condition_dict["Field"])));
    condition.set_operation(
        static_cast<serialization::ProtoBufFilterOperation>(pybind11::cast<int64_t>(condition_dict["Operation"])));
    condition.set_value(pybind11::cast<int64_t>(condition_dict["Value"]));
    return condition;
}

serialization::ProtoBufFilter parse_filter_dict(const pybind11::dict& filter_dict) {
    serialization::ProtoBufFilter filter;
    filter.set_logical_operator(static_cast<serialization::ProtoBufFilterLogicalOperator>(
        pybind11::cast<int64_t>(filter_dict["LogicalOperator"])));
    filter.set_is_leaf(pybind11::cast<bool>(filter_dict["IsLeaf"]));

    pybind11::object condition_handle = pybind11::none();
    if (filter_dict.contains("Condition")) {
        condition_handle = filter_dict["Condition"].cast<pybind11::object>();
    }
    if (!condition_handle.is_none()) {
        *filter.mutable_condition() = parse_filter_condition_dict(pybind11::cast<pybind11::dict>(condition_handle));
    }

    if (filter_dict.contains("Operands")) {
        auto operands = pybind11::cast<pybind11::iterable>(filter_dict["Operands"]);
        for (auto operand : operands) {
            *filter.add_operands() = parse_filter_object(operand.cast<pybind11::object>());
        }
    }

    return filter;
}

serialization::ProtoBufFilter parse_filter_object(const pybind11::handle& input) {
    if (pybind11::isinstance<pybind11::bytes>(input) || pybind11::isinstance<pybind11::str>(input)) {
        return parse_serialized_message<serialization::ProtoBufFilter>(input, "ProtoBufFilter");
    }
    if (pybind11::isinstance<pybind11::dict>(input)) {
        return parse_filter_dict(pybind11::cast<pybind11::dict>(input));
    }
    throw std::invalid_argument("ProtoBufFilter must be serialized protobuf bytes/string or dict");
}

std::vector<serialization::ProtoBufFilter> parse_filter_list(const pybind11::iterable& filters) {
    std::vector<serialization::ProtoBufFilter> parsed;
    for (auto item : filters) {
        parsed.push_back(parse_filter_object(item.cast<pybind11::object>()));
    }
    return parsed;
}

template <typename MessageType>
std::vector<std::vector<MessageType>> parse_nested_serialized_batch(const pybind11::iterable& batch,
                                                                    const char* message_label) {
    std::vector<std::vector<MessageType>> parsed;
    for (auto batch_item : batch) {
        auto batch_list = pybind11::cast<pybind11::list>(batch_item.cast<pybind11::object>());
        std::vector<MessageType> inner;
        inner.reserve(batch_list.size());
        for (auto item : batch_list) {
            inner.push_back(
                parse_serialized_message<MessageType>(item.cast<pybind11::object>(), message_label));
        }
        parsed.push_back(std::move(inner));
    }
    return parsed;
}

std::vector<serialization::ProtoBufCard> parse_card_batch_serialized(const pybind11::iterable& card_batch) {
    std::vector<serialization::ProtoBufCard> parsed;
    for (auto item : card_batch) {
        parsed.push_back(
            parse_serialized_message<serialization::ProtoBufCard>(item.cast<pybind11::object>(), "ProtoBufCard"));
    }
    return parsed;
}

}  // namespace

// Expose class to Python
PYBIND11_MODULE(kumpel_embedding, m) {
    pybind11::module_::import("torch");
    pybind11::class_<PositionalEmbeddingImpl, torch::nn::Module, std::shared_ptr<PositionalEmbeddingImpl>>(
        m, "PositionalEmbedding")
        .def(pybind11::init<int64_t, double, int64_t, torch::Device, torch::Dtype>(), pybind11::arg("d_model"),
             pybind11::arg("dropout") = 0.1, pybind11::arg("max_len") = 5000,
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &PositionalEmbeddingImpl::forward)
        .def("forward_packed", &PositionalEmbeddingImpl::forward_packed);
    pybind11::class_<MultiHeadAttentionImpl, torch::nn::Module, std::shared_ptr<MultiHeadAttentionImpl>>(
        m, "MultiHeadAttention")
        .def(pybind11::init<int64_t, int64_t, int64_t, int64_t, int64_t, double, bool, torch::Device, torch::Dtype>(),
             pybind11::arg("d_q"), pybind11::arg("d_k"), pybind11::arg("d_v"), pybind11::arg("d_head"),
             pybind11::arg("nheads"), pybind11::arg("dropout") = 0.0, pybind11::arg("bias") = true,
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &MultiHeadAttentionImpl::forward)
        .def("save_weights", &MultiHeadAttentionImpl::save_weights)
        .def("load_weights", &MultiHeadAttentionImpl::load_weights);
    pybind11::class_<NormalizedLinearImpl, torch::nn::Module, std::shared_ptr<NormalizedLinearImpl>>(m,
                                                                                                     "NormalizedLinear")
        .def(pybind11::init<int64_t, int64_t, double, torch::Device, torch::Dtype>(), pybind11::arg("d_in"),
             pybind11::arg("d_out"), pybind11::arg("divisor") = 400.0,
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &NormalizedLinearImpl::forward)
        .def("save_weights", &NormalizedLinearImpl::save_weights)
        .def("load_weights", &NormalizedLinearImpl::load_weights);
    pybind11::class_<SharedEmbeddingHolderImpl, torch::nn::Module, std::shared_ptr<SharedEmbeddingHolderImpl>>(
        m, "SharedEmbeddingHolder")
        .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("save_weights", &SharedEmbeddingHolderImpl::save_weights)
        .def("load_weights", &SharedEmbeddingHolderImpl::load_weights);
    pybind11::class_<AttackDataEmbeddingImpl, torch::nn::Module, std::shared_ptr<AttackDataEmbeddingImpl>>(
        m, "AttackDataEmbedding")
        .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("shared_embedding_holder"), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &AttackDataEmbeddingImpl::forward)
        .def("save_weights", &AttackDataEmbeddingImpl::save_weights)
        .def("load_weights", &AttackDataEmbeddingImpl::load_weights);
    pybind11::class_<DiscardDataEmbeddingImpl, torch::nn::Module, std::shared_ptr<DiscardDataEmbeddingImpl>>(
        m, "DiscardDataEmbedding")
        .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &DiscardDataEmbeddingImpl::forward)
        .def("save_weights", &DiscardDataEmbeddingImpl::save_weights)
        .def("load_weights", &DiscardDataEmbeddingImpl::load_weights);
    pybind11::class_<CardAmountDataEmbeddingImpl, torch::nn::Module, std::shared_ptr<CardAmountDataEmbeddingImpl>>(
        m, "CardAmountDataEmbedding")
        .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("shared_embedding_holder"), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &CardAmountDataEmbeddingImpl::forward)
        .def("save_weights", &CardAmountDataEmbeddingImpl::save_weights)
        .def("load_weights", &CardAmountDataEmbeddingImpl::load_weights);
    pybind11::class_<ReturnToDeckTypeDataEmbeddingImpl, torch::nn::Module,
                     std::shared_ptr<ReturnToDeckTypeDataEmbeddingImpl>>(m, "ReturnToDeckTypeDataEmbedding")
        .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("shared_embedding_holder"), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &ReturnToDeckTypeDataEmbeddingImpl::forward)
        .def("save_weights", &ReturnToDeckTypeDataEmbeddingImpl::save_weights)
        .def("load_weights", &ReturnToDeckTypeDataEmbeddingImpl::load_weights);
    pybind11::class_<PlayerTargetDataEmbeddingImpl, torch::nn::Module, std::shared_ptr<PlayerTargetDataEmbeddingImpl>>(
        m, "PlayerTargetDataEmbedding")
        .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("shared_embedding_holder"), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &PlayerTargetDataEmbeddingImpl::forward)
        .def("save_weights", &PlayerTargetDataEmbeddingImpl::save_weights)
        .def("load_weights", &PlayerTargetDataEmbeddingImpl::load_weights);
    pybind11::class_<FilterConditionEmbeddingImpl, torch::nn::Module, std::shared_ptr<FilterConditionEmbeddingImpl>>(
        m, "FilterConditionEmbedding")
        .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("shared_embedding_holder"), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", &FilterConditionEmbeddingImpl::forward)
        .def("save_weights", &FilterConditionEmbeddingImpl::save_weights)
        .def("load_weights", &FilterConditionEmbeddingImpl::load_weights);
    pybind11::class_<FilterEmbeddingImpl, torch::nn::Module, std::shared_ptr<FilterEmbeddingImpl>>(m, "FilterEmbedding")
        .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("shared_embedding_holder"), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward", [](FilterEmbeddingImpl& self,
                           const pybind11::iterable& filter) { return self.forward(parse_filter_list(filter)); })
        .def("save_weights", &FilterEmbeddingImpl::save_weights)
        .def("load_weights", &FilterEmbeddingImpl::load_weights);
    pybind11::class_<InstructionDataEmbeddingImpl, torch::nn::Module, std::shared_ptr<InstructionDataEmbeddingImpl>>(
        m, "InstructionDataEmbedding")
        .def(pybind11::init<std::shared_ptr<SharedEmbeddingHolderImpl>, int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("shared_embedding_holder"), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward",
             [](InstructionDataEmbeddingImpl& self, const pybind11::iterable& instructions_batch) {
                 auto parsed = parse_nested_serialized_batch<serialization::ProtoBufInstruction>(
                     instructions_batch, "ProtoBufInstruction");
                 auto flat = nesting::flatten_instructions(parsed, std::nullopt, torch::kInt64);
                 flat = nesting::move_flattened_result_to_device(flat, self.parameters()[0].device());
                 return self.forward(flat);
             })
        .def("save_weights", &InstructionDataEmbeddingImpl::save_weights)
        .def("load_weights", &InstructionDataEmbeddingImpl::load_weights);
    pybind11::class_<InstructionEmbeddingImpl, torch::nn::Module, std::shared_ptr<InstructionEmbeddingImpl>>(
        m, "InstructionEmbedding")
        .def(pybind11::init<std::shared_ptr<InstructionDataEmbeddingImpl>, std::shared_ptr<SharedEmbeddingHolderImpl>,
                            int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("instruction_data_embedding"), pybind11::arg("shared_embedding_holder"),
             pybind11::arg("dimension_out"), pybind11::arg("device") = torch::Device(torch::kCPU),
             pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward",
             [](InstructionEmbeddingImpl& self, const pybind11::iterable& instructions_batch) {
                 auto parsed = parse_nested_serialized_batch<serialization::ProtoBufInstruction>(
                     instructions_batch, "ProtoBufInstruction");
                 return self.forward(parsed);
             })
        .def("save_weights", &InstructionEmbeddingImpl::save_weights)
        .def("load_weights", &InstructionEmbeddingImpl::load_weights);
    pybind11::class_<ConditionEmbeddingImpl, torch::nn::Module, std::shared_ptr<ConditionEmbeddingImpl>>(
        m, "ConditionEmbedding")
        .def(pybind11::init<std::shared_ptr<InstructionDataEmbeddingImpl>, std::shared_ptr<SharedEmbeddingHolderImpl>,
                            int64_t, torch::Device, torch::Dtype>(),
             pybind11::arg("instruction_data_embedding"), pybind11::arg("shared_embedding_holder"),
             pybind11::arg("dimension_out"), pybind11::arg("device") = torch::Device(torch::kCPU),
             pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward",
             [](ConditionEmbeddingImpl& self, const pybind11::iterable& conditions_batch) {
                 auto parsed = parse_nested_serialized_batch<serialization::ProtoBufCondition>(
                     conditions_batch, "ProtoBufCondition");
                 return self.forward(parsed);
             })
        .def("save_weights", &ConditionEmbeddingImpl::save_weights)
        .def("load_weights", &ConditionEmbeddingImpl::load_weights);
    pybind11::class_<CardEmbeddingImpl, torch::nn::Module, std::shared_ptr<CardEmbeddingImpl>>(m, "CardEmbedding")
        .def(pybind11::init<int64_t, torch::Device, torch::Dtype>(), pybind11::arg("dimension_out"),
             pybind11::arg("device") = torch::Device(torch::kCPU), pybind11::arg("dtype") = torch::Dtype(torch::kFloat))
        .def("forward",
             [](CardEmbeddingImpl& self, const pybind11::iterable& cards) {
                 return self.forward(parse_card_batch_serialized(cards));
             })
        .def("save_weights", &CardEmbeddingImpl::save_weights)
        .def("load_weights", &CardEmbeddingImpl::load_weights);

    m.def("nesting_traverse_filter", [](const pybind11::iterable& nested_input) {
        auto nodes = parse_filter_list(nested_input);
        auto entries = nesting::traverse_filter(nodes);
        pybind11::list out;
        for (const auto& entry : entries) {
            out.append(pybind11::make_tuple(entry.value, group_index_to_py_tuple(entry.group_index), entry.op));
        }
        return out;
    });

    m.def("nesting_flatten_filter", [](const pybind11::iterable& nested_input) {
        auto nodes = parse_filter_list(nested_input);
        auto entries = nesting::traverse_filter(nodes);
        auto result = nesting::flatten(entries);

        pybind11::list groups;
        pybind11::dict operators;
        for (const auto& entry : entries) {
            auto key = group_index_to_py_tuple(entry.group_index);
            operators[key] = pybind11::int_(entry.op);
        }
        for (const auto& group : result.groups) {
            groups.append(group_index_to_py_tuple(group));
        }

        return pybind11::make_tuple(result.flattened_input, groups, operators);
    });

    m.def("nesting_reduce", [](const torch::Tensor& flattened_input, const pybind11::list& groups,
                               const pybind11::dict& operators, pybind11::function combine_function) {
        std::vector<nesting::GroupIndex> cpp_groups;
        cpp_groups.reserve(groups.size());
        for (auto item : groups) {
            cpp_groups.push_back(py_tuple_to_group_index(pybind11::cast<pybind11::tuple>(item)));
        }

        nesting::OperatorMap cpp_operators;
        for (auto item : operators) {
            auto key_tuple = pybind11::cast<pybind11::tuple>(item.first);
            cpp_operators[py_tuple_to_group_index(key_tuple)] = pybind11::cast<int64_t>(item.second);
        }

        std::vector<torch::Tensor> cpp_flattened;
        cpp_flattened.reserve(flattened_input.size(0));
        for (int64_t i = 0; i < flattened_input.size(0); ++i) {
            cpp_flattened.push_back(flattened_input[i]);
        }

        const nesting::ReduceCombineFunction cpp_combine = [&combine_function](const std::vector<torch::Tensor>& values,
                                                                               std::optional<int64_t> op) {
            pybind11::list py_values;
            for (const auto& value : values) {
                py_values.append(value);
            }
            pybind11::object py_op = op.has_value() ? pybind11::cast(*op) : pybind11::none();
            return combine_function(py_values, py_op).cast<torch::Tensor>();
        };

        auto reduced = nesting::reduce(nesting::ReduceRequest{cpp_flattened, cpp_groups, cpp_operators, cpp_combine});

        pybind11::list out;
        for (const auto& tensor : reduced) {
            out.append(tensor);
        }
        return out;
    });
}