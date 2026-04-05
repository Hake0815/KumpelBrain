#include "../include/CardEmbedding.h"

#include <vector>

namespace {

template <typename T>
std::vector<T> toVector(const google::protobuf::RepeatedPtrField<T>& proto_list) {
    return std::vector<T>(proto_list.begin(), proto_list.end());
}

}  // namespace

CardEmbeddingImpl::CardEmbeddingImpl(int64_t dimension_out, torch::Device device, torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    shared_embedding_holder_ =
        register_module("shared_embedding_holder", SharedEmbeddingHolder(dimension_out, device, dtype));
    instruction_data_embedding_ =
        register_module("instruction_data_embedding",
                        InstructionDataEmbedding(shared_embedding_holder_.ptr(), dimension_out, device, dtype));
    instruction_embedding_ = register_module(
        "instruction_embedding", InstructionEmbedding(instruction_data_embedding_.ptr(), shared_embedding_holder_.ptr(),
                                                      dimension_out, device, dtype));
    condition_embedding_ = register_module(
        "condition_embedding", ConditionEmbedding(instruction_data_embedding_.ptr(), shared_embedding_holder_.ptr(),
                                                  dimension_out, device, dtype));
    ability_embedding_ = register_module("ability_embedding", AbilityEmbedding(dimension_out, device, dtype));
    attack_embedding_ = register_module("attack_embedding", AttackEmbedding(dimension_out, device, dtype));
}

torch::Tensor CardEmbeddingImpl::forward(const std::vector<ProtoBufCard>& card_batch) {
    auto instructions_and_conditions = collect_instructions_and_conditions(card_batch);
    embed_instructions_and_conditions(instructions_and_conditions, static_cast<int>(card_batch.size()));
    return torch::empty({0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
}
InstructionsAndConditions CardEmbeddingImpl::collect_instructions_and_conditions(
    const std::vector<ProtoBufCard>& card_batch) {
    std::vector<std::vector<ProtoBufInstruction>> instructions;
    std::vector<std::vector<ProtoBufCondition>> conditions;
    std::vector<std::pair<int, int>> instruction_card_parent_indices;
    std::vector<std::pair<int, int>> condition_card_parent_indices;
    std::vector<int64_t> instruction_card_indices;
    std::vector<int64_t> instruction_ability_indices;
    std::vector<int64_t> instruction_attack_indices;
    std::vector<int64_t> condition_card_indices;
    std::vector<int64_t> condition_ability_indices;
    std::vector<torch::Tensor> attack_energy_costs;
    std::vector<std::pair<int, int>> attack_energy_cost_parent_indices;
    for (int card_index = 0; card_index < card_batch.size(); ++card_index) {
        const auto& card = card_batch[card_index];
        if (card.instructions_size() > 0) {
            instructions.push_back(toVector<>(card.instructions()));
            instruction_card_parent_indices.push_back({card_index, 0});
            instruction_card_indices.push_back(instructions.size() - 1);
        }
        if (card.conditions_size() > 0) {
            conditions.push_back(toVector<>(card.conditions()));
            condition_card_parent_indices.push_back({card_index, 0});
            condition_card_indices.push_back(conditions.size() - 1);
        }
        if (card.has_ability()) {
            auto ability = card.ability();
            if (ability.instructions_size() > 0) {
                instructions.push_back(toVector<>(ability.instructions()));
                instruction_card_parent_indices.push_back({card_index, 0});
                instruction_ability_indices.push_back(instructions.size() - 1);
            }
            if (ability.conditions_size() > 0) {
                conditions.push_back(toVector<>(ability.conditions()));
                condition_card_parent_indices.push_back({card_index, 0});
                condition_ability_indices.push_back(conditions.size() - 1);
            }
        }
        if (card.attacks_size() > 0) {
            for (int attack_index = 0; attack_index < card.attacks_size(); ++attack_index) {
                auto attack = card.attacks(attack_index);
                if (attack.instructions_size() > 0) {
                    if (attack.energy_cost_size() > 0) {
                        auto attack_energy_cost_tensor = torch::tensor(
                            {attack.energy_cost()}, torch::TensorOptions().device(device_).dtype(torch::kLong));
                        attack_energy_costs.push_back(
                            shared_embedding_holder_->energy_type_embedding_(attack_energy_cost_tensor).sum(0));
                    } else {
                        attack_energy_costs.push_back(
                            torch::zeros({dimension_out_}, torch::TensorOptions().device(device_).dtype(torch::kLong)));
                    }
                    attack_energy_cost_parent_indices.push_back({card_index, attack_index});
                    instructions.push_back(toVector<>(attack.instructions()));
                    instruction_card_parent_indices.push_back({card_index, attack_index});
                    instruction_attack_indices.push_back(instructions.size() - 1);
                }
            }
        }
    }
    return InstructionsAndConditions{instructions,
                                     conditions,
                                     instruction_card_parent_indices,
                                     condition_card_parent_indices,
                                     instruction_card_indices,
                                     instruction_ability_indices,
                                     instruction_attack_indices,
                                     condition_card_indices,
                                     condition_ability_indices,
                                     attack_energy_costs,
                                     attack_energy_cost_parent_indices};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_instructions_and_conditions(
    const InstructionsAndConditions& instructions_and_conditions, int batch_size) {
    auto embedded_instructions_pair = instruction_embedding_->forward(instructions_and_conditions.instructions);
    auto embedded_conditions_pair = condition_embedding_->forward(instructions_and_conditions.conditions);

    return embed_attacks(embedded_instructions_pair, instructions_and_conditions.instruction_attack_indices,
                         instructions_and_conditions.attack_energy_costs,
                         instructions_and_conditions.instruction_card_parent_indices, batch_size);
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_attacks(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_attack_indices, const std::vector<torch::Tensor>& attack_energy_costs,
    const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size) {
    auto embedded_instruction_attacks =
        embedded_instructions_pair.first.index_select(0, torch::tensor(instruction_attack_indices, torch::kLong));
    auto embedded_instruction_attacks_mask =
        embedded_instructions_pair.second.index_select(0, torch::tensor(instruction_attack_indices, torch::kLong));
    auto embedded_attacks = attack_embedding_->forward(torch::stack(attack_energy_costs), embedded_instruction_attacks,
                                                       embedded_instruction_attacks_mask);
    // Gather max_attacks
    int max_attacks = 0;
    for (int instruction_attack_index : instruction_attack_indices) {
        int number_of_attacks = instruction_card_parent_indices[instruction_attack_index].second + 1;
        if (number_of_attacks > max_attacks) max_attacks = number_of_attacks;
    }
    auto device = embedded_attacks.device();
    int64_t dimension_out = embedded_attacks.size(1);

    // Prepare output and mask
    auto out_options = torch::TensorOptions().device(device).dtype(embedded_attacks.dtype());
    auto mask_options = torch::TensorOptions().device(device).dtype(torch::kBool);
    auto out = torch::zeros({batch_size, max_attacks, dimension_out}, out_options);
    auto mask = torch::zeros({batch_size, max_attacks}, mask_options);

    for (int i = 0; i < instruction_attack_indices.size(); ++i) {
        auto [card_index, attack_pos] = instruction_card_parent_indices[instruction_attack_indices[i]];
        out[card_index][attack_pos] = embedded_attacks[i];
        mask[card_index][attack_pos] = true;
    }
    return {out, mask};
}