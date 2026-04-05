#include "../include/CardEmbedding.h"

#include <vector>

#include "network/include/AttentionUtils.h"

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
    card_instructions_multi_head_attention_ =
        register_module("card_instructions_multi_head_attention",
                        MultiHeadAttention(dimension_out, dimension_out, dimension_out,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0, true, device, dtype));
    card_conditions_multi_head_attention_ =
        register_module("card_conditions_multi_head_attention",
                        MultiHeadAttention(dimension_out, dimension_out, dimension_out,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 4, 0.0, true, device, dtype));
    card_instruction_query_embedding_ =
        register_module("card_instruction_query_embedding", torch::nn::Embedding(1, dimension_out));
    card_condition_query_embedding_ =
        register_module("card_condition_query_embedding", torch::nn::Embedding(1, dimension_out));
    card_token_multi_head_attention_ =
        register_module("card_token_multi_head_attention",
                        MultiHeadAttention(dimension_out, dimension_out, dimension_out,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 8, 0.0, true, device, dtype));
    card_token_query_embedding_ = register_module("card_token_query_embedding", torch::nn::Embedding(1, dimension_out));
    to(device, dtype);
}

torch::Tensor CardEmbeddingImpl::forward(const std::vector<ProtoBufCard>& card_batch) {
    auto instructions_and_conditions = collect_instructions_and_conditions(card_batch);
    auto batch_size = static_cast<int>(card_batch.size());
    auto [tokens, mask] = embed_instructions_and_conditions(instructions_and_conditions, batch_size);

    auto query = card_token_query_embedding_
                     ->forward(torch::zeros({batch_size}, torch::TensorOptions().device(device_).dtype(torch::kLong)))
                     .unsqueeze(1);
    return attention_utils::masked_attention_pooling(card_token_multi_head_attention_, query, tokens, mask);
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
                        auto attack_energy_cost_vector =
                            std::vector<int64_t>(attack.energy_cost().begin(), attack.energy_cost().end());
                        auto attack_energy_cost_tensor = torch::tensor(
                            attack_energy_cost_vector, torch::TensorOptions().device(device_).dtype(torch::kLong));
                        attack_energy_costs.push_back(
                            shared_embedding_holder_->energy_type_embedding_(attack_energy_cost_tensor).sum(0));
                    } else {
                        attack_energy_costs.push_back(
                            torch::zeros({dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_)));
                    }
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
                                     attack_energy_costs};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_instructions_and_conditions(
    const InstructionsAndConditions& instructions_and_conditions, int batch_size) {
    auto embedded_instructions_pair = instruction_embedding_->forward(instructions_and_conditions.instructions);
    auto embedded_conditions_pair = condition_embedding_->forward(instructions_and_conditions.conditions);

    auto [embedded_attacks, mask_attacks] =
        embed_attacks(embedded_instructions_pair, instructions_and_conditions.instruction_attack_indices,
                      instructions_and_conditions.attack_energy_costs,
                      instructions_and_conditions.instruction_card_parent_indices, batch_size);
    auto [embedded_abilities, mask_abilities] =
        embed_ability(embedded_instructions_pair, instructions_and_conditions.instruction_ability_indices,
                      embedded_conditions_pair, instructions_and_conditions.condition_ability_indices,
                      instructions_and_conditions.instruction_card_parent_indices, batch_size);
    auto [embedded_card_instructions, mask_card_instructions] =
        embed_card_instructions(embedded_instructions_pair, instructions_and_conditions.instruction_card_indices,
                                instructions_and_conditions.instruction_card_parent_indices, batch_size);
    auto [embedded_card_conditions, mask_card_conditions] =
        embed_card_conditions(embedded_conditions_pair, instructions_and_conditions.condition_card_indices,
                              instructions_and_conditions.condition_card_parent_indices, batch_size);
    return {torch::cat({embedded_card_instructions, embedded_card_conditions, embedded_abilities, embedded_attacks}, 1),
            torch::cat({mask_card_instructions, mask_card_conditions, mask_abilities, mask_attacks}, 1)};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_attacks(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_attack_indices, const std::vector<torch::Tensor>& attack_energy_costs,
    const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size) {
    if (attack_energy_costs.empty()) {
        return {torch::zeros({batch_size, 0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_)),
                torch::zeros({batch_size, 0}, torch::TensorOptions().device(device_).dtype(torch::kBool))};
    }
    auto tensorOptions = torch::TensorOptions().device(device_).dtype(torch::kLong);
    auto embedded_instruction_attacks =
        embedded_instructions_pair.first.index_select(0, torch::tensor(instruction_attack_indices, tensorOptions));
    auto embedded_instruction_attacks_mask =
        embedded_instructions_pair.second.index_select(0, torch::tensor(instruction_attack_indices, tensorOptions));
    auto embedded_attacks = attack_embedding_->forward(torch::stack(attack_energy_costs), embedded_instruction_attacks,
                                                       embedded_instruction_attacks_mask);
    // Gather max_attacks
    int max_attacks = 0;
    for (int instruction_attack_index : instruction_attack_indices) {
        int number_of_attacks = instruction_card_parent_indices[instruction_attack_index].second + 1;
        if (number_of_attacks > max_attacks) max_attacks = number_of_attacks;
    }

    // Prepare output and mask
    auto out_options = torch::TensorOptions().device(device_).dtype(dtype_);
    auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    auto out = torch::zeros({batch_size, max_attacks, dimension_out_}, out_options);
    auto mask = torch::zeros({batch_size, max_attacks}, mask_options);

    for (int i = 0; i < instruction_attack_indices.size(); ++i) {
        auto [card_index, attack_pos] = instruction_card_parent_indices[instruction_attack_indices[i]];
        out[card_index][attack_pos] = embedded_attacks[i];
        mask[card_index][attack_pos] = true;
    }
    return {out, mask};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_ability(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_ability_indices,
    const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
    const std::vector<int64_t>& condition_ability_indices,
    const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size) {
    auto tensorOptions = torch::TensorOptions().device(device_).dtype(torch::kLong);
    auto embedded_instruction_abilities =
        embedded_instructions_pair.first.index_select(0, torch::tensor(instruction_ability_indices, tensorOptions));
    auto embedded_instruction_abilities_mask =
        embedded_instructions_pair.second.index_select(0, torch::tensor(instruction_ability_indices, tensorOptions));
    auto embedded_instruction_conditions =
        embedded_conditions_pair.first.index_select(0, torch::tensor(condition_ability_indices, tensorOptions));
    auto embedded_instruction_conditions_mask =
        embedded_conditions_pair.second.index_select(0, torch::tensor(condition_ability_indices, tensorOptions));
    auto embedded_abilities =
        ability_embedding_->forward(embedded_instruction_abilities, embedded_instruction_abilities_mask,
                                    embedded_instruction_conditions, embedded_instruction_conditions_mask);

    // ability must have instructions but may have conditions
    return pad_to_batch(instruction_ability_indices, instruction_card_parent_indices, batch_size, embedded_abilities);
}
std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_instructions(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_card_indices,
    const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size) {
    auto tensorOptions = torch::TensorOptions().device(device_).dtype(torch::kLong);
    auto embedded_instructions =
        embedded_instructions_pair.first.index_select(0, torch::tensor(instruction_card_indices, tensorOptions));

    auto embedded_instructions_mask =
        embedded_instructions_pair.second.index_select(0, torch::tensor(instruction_card_indices, tensorOptions));

    auto instruction_query =
        card_instruction_query_embedding_->forward(torch::zeros({embedded_instructions.size(0)}, tensorOptions))
            .unsqueeze(1);

    auto pooled_instructions = attention_utils::masked_attention_pooling(
        card_instructions_multi_head_attention_, instruction_query, embedded_instructions, embedded_instructions_mask);

    return pad_to_batch(instruction_card_indices, instruction_card_parent_indices, batch_size, pooled_instructions);
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_conditions(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
    const std::vector<int64_t>& condition_card_indices,
    const std::vector<std::pair<int, int>>& condition_card_parent_indices, int batch_size) {
    auto tensorOptions = torch::TensorOptions().device(device_).dtype(torch::kLong);
    auto embedded_conditions =
        embedded_conditions_pair.first.index_select(0, torch::tensor(condition_card_indices, tensorOptions));

    auto embedded_conditions_mask =
        embedded_conditions_pair.second.index_select(0, torch::tensor(condition_card_indices, tensorOptions));

    auto condition_query =
        card_condition_query_embedding_->forward(torch::zeros({embedded_conditions.size(0)}, tensorOptions))
            .unsqueeze(1);

    auto pooled_conditions = attention_utils::masked_attention_pooling(
        card_conditions_multi_head_attention_, condition_query, embedded_conditions, embedded_conditions_mask);

    return pad_to_batch(condition_card_indices, condition_card_parent_indices, batch_size, pooled_conditions);
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::pad_to_batch(
    const std::vector<int64_t>& card_indices, const std::vector<std::pair<int, int>>& card_parent_indices,
    int batch_size, const torch::Tensor& pooled_tokens) {
    auto out_options = torch::TensorOptions().device(device_).dtype(dtype_);
    auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    auto out = torch::zeros({batch_size, dimension_out_}, out_options);
    auto mask = torch::zeros({batch_size}, mask_options);

    for (int i = 0; i < card_indices.size(); ++i) {
        int card_index = card_parent_indices[card_indices[i]].first;
        out[card_index] = pooled_tokens[i];
        mask[card_index] = true;
    }
    return {out.unsqueeze(1), mask.unsqueeze(1)};
}