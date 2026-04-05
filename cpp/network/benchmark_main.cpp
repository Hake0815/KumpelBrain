#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <torch/cuda.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../network/include/ConditionEmbedding.h"
#include "../network/include/InstructionDataEmbedding.h"
#include "../network/include/InstructionEmbedding.h"
#include "../network/include/Nesting.h"
#include "../network/include/SharedEmbeddingHolder.h"

namespace serialization = gamecore::serialization;

namespace {

volatile int64_t benchmark_sink = 0;

serialization::ProtoBufFilter make_leaf_filter(int field, int operation, int value) {
    serialization::ProtoBufFilter filter;
    filter.set_is_leaf(true);
    filter.set_logical_operator(serialization::FILTER_LOGICAL_OPERATOR_NONE);
    auto* condition = filter.mutable_condition();
    condition->set_field(static_cast<serialization::ProtoBufFilterType>(field));
    condition->set_operation(static_cast<serialization::ProtoBufFilterOperation>(operation));
    condition->set_value(value);
    return filter;
}

serialization::ProtoBufFilter make_group_filter(serialization::ProtoBufFilterLogicalOperator logical_operator,
                                                const std::vector<serialization::ProtoBufFilter>& operands) {
    serialization::ProtoBufFilter filter;
    filter.set_is_leaf(false);
    filter.set_logical_operator(logical_operator);
    for (const auto& operand : operands) {
        *filter.add_operands() = operand;
    }
    return filter;
}

serialization::ProtoBufFilter make_nested_filter(int64_t batch_index, int64_t instruction_index) {
    const auto card_type = static_cast<int>((batch_index + instruction_index) % 4);
    const auto card_subtype = static_cast<int>(1 + ((batch_index + instruction_index) % 8));
    const auto hp_threshold = static_cast<int>(40 + ((batch_index * 7 + instruction_index * 11) % 220));

    return make_group_filter(
        serialization::FILTER_LOGICAL_OPERATOR_OR,
        {
            make_group_filter(serialization::FILTER_LOGICAL_OPERATOR_AND,
                              {
                                  make_leaf_filter(serialization::FILTER_TYPE_CARD_TYPE,
                                                   serialization::FILTER_OPERATION_EQUALS, card_type),
                                  make_leaf_filter(serialization::FILTER_TYPE_HP,
                                                   serialization::FILTER_OPERATION_GREATER_THAN_OR_EQUAL, hp_threshold),
                              }),
            make_leaf_filter(serialization::FILTER_TYPE_CARD_SUBTYPE, serialization::FILTER_OPERATION_EQUALS,
                             card_subtype),
        });
}

serialization::ProtoBufInstructionData make_attack_data(int damage) {
    serialization::ProtoBufInstructionData data;
    data.set_instruction_data_type(serialization::INSTRUCTION_DATA_TYPE_ATTACK_DATA);
    auto* attack = data.mutable_attack_data();
    attack->set_attack_target(serialization::ATTACK_TARGET_DEFENDING_POKEMON);
    attack->set_damage(damage);
    return data;
}

serialization::ProtoBufInstructionData make_discard_data(int source) {
    serialization::ProtoBufInstructionData data;
    data.set_instruction_data_type(serialization::INSTRUCTION_DATA_TYPE_DISCARD_DATA);
    data.mutable_discard_data()->set_target_source(static_cast<serialization::ProtoBufTargetSource>(source));
    return data;
}

serialization::ProtoBufInstructionData make_card_amount_data(int min_amount, int max_amount, int from_position) {
    serialization::ProtoBufInstructionData data;
    data.set_instruction_data_type(serialization::INSTRUCTION_DATA_TYPE_CARD_AMOUNT_DATA);
    auto* amount_data = data.mutable_card_amount_data();
    amount_data->mutable_amount()->set_min(min_amount);
    amount_data->mutable_amount()->set_max(max_amount);
    amount_data->set_from_position(static_cast<serialization::ProtoBufCardPosition>(from_position));
    return data;
}

serialization::ProtoBufInstructionData make_return_to_deck_type_data(int return_type, int from_position) {
    serialization::ProtoBufInstructionData data;
    data.set_instruction_data_type(serialization::INSTRUCTION_DATA_TYPE_RETURN_TO_DECK_TYPE_DATA);
    auto* return_data = data.mutable_return_to_deck_type_data();
    return_data->set_return_to_deck_type(static_cast<serialization::ProtoBufReturnToDeckType>(return_type));
    return_data->set_from_position(static_cast<serialization::ProtoBufCardPosition>(from_position));
    return data;
}

serialization::ProtoBufInstructionData make_filter_data(const serialization::ProtoBufFilter& filter) {
    serialization::ProtoBufInstructionData data;
    data.set_instruction_data_type(serialization::INSTRUCTION_DATA_TYPE_FILTER_DATA);
    *data.mutable_filter_data()->mutable_filter() = filter;
    return data;
}

serialization::ProtoBufInstructionData make_player_target_data(int target) {
    serialization::ProtoBufInstructionData data;
    data.set_instruction_data_type(serialization::INSTRUCTION_DATA_TYPE_PLAYER_TARGET_DATA);
    data.mutable_player_target_data()->set_player_target(static_cast<serialization::ProtoBufPlayerTarget>(target));
    return data;
}

serialization::ProtoBufInstruction make_instruction(int instruction_type,
                                                    const std::vector<serialization::ProtoBufInstructionData>& data) {
    serialization::ProtoBufInstruction instruction;
    instruction.set_instruction_type(static_cast<serialization::ProtoBufInstructionType>(instruction_type));
    for (const auto& entry : data) {
        *instruction.add_data() = entry;
    }
    return instruction;
}

serialization::ProtoBufCondition make_condition(int condition_type,
                                                const std::vector<serialization::ProtoBufInstructionData>& data) {
    serialization::ProtoBufCondition condition;
    condition.set_condition_type(static_cast<serialization::ProtoBufConditionType>(condition_type));
    for (const auto& entry : data) {
        *condition.add_data() = entry;
    }
    return condition;
}

std::vector<std::vector<serialization::ProtoBufInstruction>> build_instruction_batches(int64_t batch_size,
                                                                                       int64_t instructions_per_batch) {
    std::vector<std::vector<serialization::ProtoBufInstruction>> batches;
    batches.reserve(batch_size);

    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        std::vector<serialization::ProtoBufInstruction> batch;
        batch.reserve(instructions_per_batch);

        for (int64_t instruction_index = 0; instruction_index < instructions_per_batch; ++instruction_index) {
            switch (instruction_index % 8) {
                case 0:
                    batch.push_back(make_instruction(
                        serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                        {make_attack_data(20 + static_cast<int>((batch_index + instruction_index) % 120))}));
                    break;
                case 1:
                    batch.push_back(
                        make_instruction(serialization::INSTRUCTION_TYPE_SELECT_CARDS,
                                         {make_card_amount_data(1, 2, serialization::CARD_POSITION_HAND),
                                          make_filter_data(make_nested_filter(batch_index, instruction_index))}));
                    break;
                case 2:
                    batch.push_back(
                        make_instruction(serialization::INSTRUCTION_TYPE_DISCARD,
                                         {make_discard_data(static_cast<int>((batch_index + instruction_index) % 3))}));
                    break;
                case 3:
                    batch.push_back(make_instruction(serialization::INSTRUCTION_TYPE_TAKE_TO_HAND,
                                                     {make_card_amount_data(1, 1, serialization::CARD_POSITION_DECK)}));
                    break;
                case 4:
                    batch.push_back(make_instruction(
                        serialization::INSTRUCTION_TYPE_PUT_IN_DECK,
                        {make_return_to_deck_type_data(static_cast<int>((batch_index + instruction_index) % 2),
                                                       serialization::CARD_POSITION_DISCARD_PILE)}));
                    break;
                case 5:
                    batch.push_back(make_instruction(
                        serialization::INSTRUCTION_TYPE_REVEAL_CARDS,
                        {make_card_amount_data(1, 3, serialization::CARD_POSITION_SELECTED_CARDS),
                         make_filter_data(make_leaf_filter(serialization::FILTER_TYPE_EXCLUDE_SOURCE,
                                                           serialization::FILTER_OPERATION_NONE, 0))}));
                    break;
                case 6:
                    batch.push_back(make_instruction(serialization::INSTRUCTION_TYPE_SHOW_CARDS, {}));
                    break;
                default:
                    batch.push_back(make_instruction(
                        serialization::INSTRUCTION_TYPE_SHUFFLE_DECK,
                        {make_player_target_data(static_cast<int>((batch_index + instruction_index) % 2))}));
                    break;
            }
        }

        batches.push_back(std::move(batch));
    }

    return batches;
}

std::vector<std::vector<serialization::ProtoBufCondition>> build_condition_batches(int64_t batch_size,
                                                                                   int64_t conditions_per_batch) {
    std::vector<std::vector<serialization::ProtoBufCondition>> batches;
    batches.reserve(batch_size);

    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        std::vector<serialization::ProtoBufCondition> batch;
        batch.reserve(conditions_per_batch);

        for (int64_t condition_index = 0; condition_index < conditions_per_batch; ++condition_index) {
            if (condition_index % 2 == 0) {
                batch.push_back(make_condition(serialization::CONDITION_TYPE_HAS_CARDS,
                                               {make_card_amount_data(1, 60, serialization::CARD_POSITION_DECK),
                                                make_filter_data(make_nested_filter(batch_index, condition_index))}));
            } else {
                batch.push_back(make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {}));
            }
        }

        batches.push_back(std::move(batch));
    }

    return batches;
}

void synchronize_device(const torch::Device& device) {
    if (device.is_cuda()) {
        torch::cuda::synchronize(device.index());
    }
}

double benchmark_ms(const std::string& name, const torch::Device& device, int warmup_runs, int measured_runs,
                    const std::function<void()>& fn) {
    for (int run = 0; run < warmup_runs; ++run) {
        fn();
    }

    synchronize_device(device);
    const auto start = std::chrono::steady_clock::now();
    for (int run = 0; run < measured_runs; ++run) {
        fn();
    }
    synchronize_device(device);
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const auto average_ms = elapsed_ms / static_cast<double>(measured_runs);
    std::cout << name << ": " << average_ms << " ms\n";
    return average_ms;
}

template <typename Fn>
void with_deterministic_algorithms(bool enabled, const Fn& fn) {
    auto& context = at::globalContext();
    const auto previous_enabled = context.deterministicAlgorithms();
    const auto previous_warn_only = context.deterministicAlgorithmsWarnOnly();
    context.setDeterministicAlgorithms(enabled, false);
    try {
        fn();
    } catch (...) {
        context.setDeterministicAlgorithms(previous_enabled, previous_warn_only);
        throw;
    }
    context.setDeterministicAlgorithms(previous_enabled, previous_warn_only);
}

void run_embedding_benchmarks(const torch::Device& device, const std::string& label) {
    const auto dtype = torch::kFloat;
    constexpr int64_t dimension = 32;
    constexpr int warmup_runs = 10;
    constexpr int measured_runs = 50;

    torch::manual_seed(42);
    if (device.is_cuda()) {
        torch::cuda::manual_seed_all(42);
    }

    auto instructions = build_instruction_batches(256, 8);
    auto conditions = build_condition_batches(256, 4);

    auto shared = std::make_shared<SharedEmbeddingHolderImpl>(dimension, device, dtype);
    auto instruction_data_embedding = std::make_shared<InstructionDataEmbeddingImpl>(shared, dimension, device, dtype);
    auto instruction_embedding =
        std::make_shared<InstructionEmbeddingImpl>(instruction_data_embedding, shared, dimension, device, dtype);
    auto condition_embedding =
        std::make_shared<ConditionEmbeddingImpl>(instruction_data_embedding, shared, dimension, device, dtype);

    shared->eval();
    instruction_data_embedding->eval();
    instruction_embedding->eval();
    condition_embedding->eval();

    const auto instruction_batch_size = static_cast<int64_t>(instructions.size());
    const auto condition_batch_size = static_cast<int64_t>(conditions.size());

    auto flat_instructions = nesting::flatten_instructions(instructions, torch::Device(torch::kCPU), torch::kInt64);
    flat_instructions = nesting::move_flattened_result_to_device(flat_instructions, device);
    auto flat_conditions = nesting::flatten_conditions(conditions, torch::Device(torch::kCPU), torch::kInt64);
    flat_conditions = nesting::move_flattened_result_to_device(flat_conditions, device);

    std::cout << "\n== " << label << " ==\n";
    std::cout << "Instruction batches: " << instruction_batch_size
              << ", flattened instructions: " << flat_instructions.instruction_indices.size(0)
              << ", flattened data rows: " << flat_instructions.instruction_data_type_indices.size(0) << "\n";
    std::cout << "Condition batches: " << condition_batch_size
              << ", flattened conditions: " << flat_conditions.instruction_indices.size(0)
              << ", flattened data rows: " << flat_conditions.instruction_data_type_indices.size(0) << "\n";

    if (device.is_cpu()) {
        benchmark_ms(label + " flatten_instructions", device, warmup_runs, measured_runs, [&]() {
            auto flat = nesting::flatten_instructions(instructions, device, torch::kInt64);
            benchmark_sink += flat.instruction_indices.size(0);
        });
    }

    benchmark_ms(label + " instruction_forward", device, warmup_runs, measured_runs, [&]() {
        auto [embedded, mask] = instruction_embedding->forward(instructions);
        benchmark_sink += embedded.numel() + mask.numel();
    });

    if (device.is_cpu()) {
        benchmark_ms(label + " flatten_conditions", device, warmup_runs, measured_runs, [&]() {
            auto flat = nesting::flatten_conditions(conditions, device, torch::kInt64);
            benchmark_sink += flat.instruction_indices.size(0);
        });
    }

    benchmark_ms(label + " condition_forward", device, warmup_runs, measured_runs, [&]() {
        auto [embedded, mask] = condition_embedding->forward(conditions);
        benchmark_sink += embedded.numel() + mask.numel();
    });
}

}  // namespace

int main() {
    torch::InferenceMode guard;
    run_embedding_benchmarks(torch::Device(torch::kCPU), "cpu");
    if (torch::cuda::is_available()) {
        run_embedding_benchmarks(torch::Device(torch::kCUDA), "cuda");
        with_deterministic_algorithms(
            true, [&]() { run_embedding_benchmarks(torch::Device(torch::kCUDA), "cuda_deterministic"); });
    } else {
        std::cout << "\nCUDA benchmark skipped: CUDA is not available.\n";
    }

    std::cout << "Benchmark sink: " << benchmark_sink << "\n";
    return 0;
}
