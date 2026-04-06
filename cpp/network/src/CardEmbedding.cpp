#include "../include/CardEmbedding.h"

#include <algorithm>
#include <vector>

#include "network/include/AttentionUtils.h"

namespace {

template <typename T>
std::vector<T> toVector(const google::protobuf::RepeatedPtrField<T>& proto_list) {
    return std::vector<T>(proto_list.begin(), proto_list.end());
}

torch::Tensor int64_vector_to_tensor(const std::vector<int64_t>& values, const torch::Device& device) {
    if (values.empty()) {
        return torch::empty({0}, torch::TensorOptions().device(device).dtype(torch::kInt64));
    }
    return torch::tensor(values, torch::TensorOptions().device(device).dtype(torch::kInt64));
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
    card_pooling_multi_head_attention_ =
        register_module("card_token_multi_head_attention",
                        MultiHeadAttention(dimension_out, dimension_out, dimension_out,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 8, 0.0, true, device, dtype));
    card_pooling_query_embedding_ =
        register_module("card_token_query_embedding", torch::nn::Embedding(1, dimension_out));
    retreat_cost_embedding_ =
        register_module("retreat_cost_embedding", NormalizedLinear(1, dimension_out, 10.0, device, dtype));
    number_of_prize_cards_on_knockout_embedding_ = register_module(
        "number_of_prize_cards_on_knockout_embedding", NormalizedLinear(1, dimension_out, 6.0, device, dtype));
    current_damage_embedding_ =
        register_module("current_damage_embedding", NormalizedLinear(1, dimension_out, 400.0, device, dtype));
    pokemon_turn_trait_embedding_ =
        register_module("pokemon_turn_trait_embedding", torch::nn::Embedding(2, dimension_out));
    card_self_multi_head_attention_ =
        register_module("card_self_multi_head_attention",
                        MultiHeadAttention(dimension_out, dimension_out, dimension_out,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 8, 0.0, true, device, dtype));
    to(device, dtype);
}

torch::Tensor CardEmbeddingImpl::forward(const std::vector<ProtoBufCard>& card_batch) {
    auto batch_size = static_cast<int>(card_batch.size());
    auto card_features = collect_card_features(card_batch);
    auto [tokens, mask] = embed_card_features(card_features, batch_size);

    auto self_attended = attention_utils::masked_self_attention(card_self_multi_head_attention_, tokens, mask);

    auto query = card_pooling_query_embedding_
                     ->forward(torch::zeros({batch_size}, torch::TensorOptions().device(device_).dtype(torch::kLong)))
                     .unsqueeze(1);
    return attention_utils::masked_attention_pooling(card_pooling_multi_head_attention_, query, self_attended, mask);
}

void CardEmbeddingImpl::collect_instructions_and_conditions(const std::vector<ProtoBufCard>& card_batch,
                                                            InstructionsAndConditions& instructions_and_conditions,
                                                            int card_index) {
    const auto& card = card_batch[card_index];
    if (card.instructions_size() > 0) {
        instructions_and_conditions.instructions.push_back(toVector<>(card.instructions()));
        instructions_and_conditions.instruction_card_parent_indices.push_back({card_index, 0});
        instructions_and_conditions.instruction_card_indices.push_back(instructions_and_conditions.instructions.size() -
                                                                       1);
    }
    if (card.conditions_size() > 0) {
        instructions_and_conditions.conditions.push_back(toVector<>(card.conditions()));
        instructions_and_conditions.condition_card_parent_indices.push_back({card_index, 0});
        instructions_and_conditions.condition_card_indices.push_back(instructions_and_conditions.conditions.size() - 1);
    }
    if (card.has_ability()) {
        auto ability = card.ability();
        int64_t ability_condition_row = -1;
        if (ability.conditions_size() > 0) {
            instructions_and_conditions.conditions.push_back(toVector<>(ability.conditions()));
            instructions_and_conditions.condition_card_parent_indices.push_back({card_index, 0});
            ability_condition_row = static_cast<int64_t>(instructions_and_conditions.conditions.size() - 1);
        }
        if (ability.instructions_size() > 0) {
            instructions_and_conditions.instructions.push_back(toVector<>(ability.instructions()));
            instructions_and_conditions.instruction_card_parent_indices.push_back({card_index, 0});
            instructions_and_conditions.instruction_ability_indices.push_back(
                instructions_and_conditions.instructions.size() - 1);
            instructions_and_conditions.ability_condition_row_for_instruction_ability.push_back(ability_condition_row);
        }
    }
    if (card.attacks_size() > 0) {
        for (int attack_index = 0; attack_index < card.attacks_size(); ++attack_index) {
            auto attack = card.attacks(attack_index);
            if (attack.instructions_size() > 0) {
                const int64_t attack_slot =
                    static_cast<int64_t>(instructions_and_conditions.instruction_attack_indices.size());
                if (attack.energy_cost_size() > 0) {
                    for (const auto energy_type : attack.energy_cost()) {
                        instructions_and_conditions.energy_flat.push_back(static_cast<int64_t>(energy_type));
                        instructions_and_conditions.energy_slot_per_token.push_back(attack_slot);
                    }
                }
                instructions_and_conditions.instructions.push_back(toVector<>(attack.instructions()));
                instructions_and_conditions.instruction_card_parent_indices.push_back({card_index, attack_index});
                instructions_and_conditions.instruction_attack_indices.push_back(
                    instructions_and_conditions.instructions.size() - 1);
            }
        }
    }
}

CardFeatures CardEmbeddingImpl::collect_card_features(const std::vector<ProtoBufCard>& card_batch) {
    CardFeatures card_features;
    for (int card_index = 0; card_index < card_batch.size(); ++card_index) {
        collect_instructions_and_conditions(card_batch, card_features.instructions_and_conditions, card_index);
        const auto& card = card_batch[card_index];
        card_features.card_type.push_back(static_cast<int64_t>(card.card_type()));

        card_features.card_subtype.push_back(static_cast<int64_t>(card.card_subtype()));

        card_features.energy_type.push_back(static_cast<int64_t>(card.energy_type()));
        card_features.energy_type_mask.push_back(static_cast<uint8_t>(card.has_energy_type()));

        card_features.max_hp.push_back(static_cast<int64_t>(card.max_hp()));
        card_features.max_hp_mask.push_back(static_cast<uint8_t>(card.has_max_hp()));

        card_features.weakness.push_back(static_cast<int64_t>(card.weakness()));
        card_features.weakness_mask.push_back(static_cast<uint8_t>(card.has_weakness()));

        card_features.resistance.push_back(static_cast<int64_t>(card.resistance()));
        card_features.resistance_mask.push_back(static_cast<uint8_t>(card.has_resistance()));

        card_features.retreat_cost.push_back(static_cast<int64_t>(card.retreat_cost()));
        card_features.retreat_cost_mask.push_back(static_cast<uint8_t>(card.has_retreat_cost()));

        card_features.number_of_prize_cards_on_knockout.push_back(
            static_cast<int64_t>(card.number_of_prize_cards_on_knockout()));
        card_features.number_of_prize_cards_on_knockout_mask.push_back(
            static_cast<uint8_t>(card.has_number_of_prize_cards_on_knockout()));

        card_features.current_damage.push_back(static_cast<int64_t>(card.current_damage()));
        card_features.current_damage_mask.push_back(static_cast<uint8_t>(card.has_current_damage()));

        if (card.pokemon_turn_traits_size() > 0) {
            for (const auto& pokemon_turn_trait : card.pokemon_turn_traits()) {
                card_features.flattened_pokemon_turn_traits.push_back(static_cast<int64_t>(pokemon_turn_trait));
                card_features.pokemon_turn_trait_card_indices.push_back(card_index);
            }
        }

        if (card.provided_energy_size() > 0) {
            for (const auto& provided_energy : card.provided_energy()) {
                card_features.flattened_provided_energies.push_back(static_cast<int64_t>(provided_energy));
                card_features.provided_energy_card_indices.push_back(card_index);
            }
        }

        if (card.attached_energy_size() > 0) {
            for (const auto& attached_energy : card.attached_energy()) {
                card_features.flattened_attached_energies.push_back(static_cast<int64_t>(attached_energy));
                card_features.attached_energy_card_indices.push_back(card_index);
            }
        }
    }

    return card_features;
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_features(const CardFeatures& card_features,
                                                                               int batch_size) {
    auto [instructions_and_conditions_tokens, instructions_and_conditions_mask] =
        embed_instructions_and_conditions(card_features.instructions_and_conditions, batch_size);

    auto mask_tensor_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    auto index_tensor_options = torch::TensorOptions().device(device_).dtype(torch::kInt64);

    auto card_type_mask = torch::ones({batch_size, 1}, mask_tensor_options);
    auto card_type_tokens =
        shared_embedding_holder_->card_type_embedding_(torch::tensor(card_features.card_type, index_tensor_options))
            .unsqueeze(1);

    auto card_subtype_mask = torch::ones({batch_size, 1}, mask_tensor_options);
    auto card_subtype_tokens =
        shared_embedding_holder_
            ->card_subtype_embedding_(torch::tensor(card_features.card_subtype, index_tensor_options))
            .unsqueeze(1);

    auto energy_type_mask = torch::tensor(card_features.energy_type_mask, mask_tensor_options).unsqueeze(1);
    auto energy_type_tokens = (shared_embedding_holder_->energy_type_embedding_(
                                   torch::tensor(card_features.energy_type, index_tensor_options)) *
                               energy_type_mask)
                                  .unsqueeze(1);

    auto max_hp_mask = torch::tensor(card_features.max_hp_mask, mask_tensor_options).unsqueeze(1);
    auto max_hp_tokens = (shared_embedding_holder_->hp_embedding_(
                              torch::tensor(card_features.max_hp, index_tensor_options).unsqueeze(-1)) *
                          max_hp_mask)
                             .unsqueeze(1);

    auto weakness_mask = torch::tensor(card_features.weakness_mask, mask_tensor_options).unsqueeze(1);
    auto weakness_tokens =
        (shared_embedding_holder_->energy_type_embedding_(torch::tensor(card_features.weakness, index_tensor_options)) *
         weakness_mask)
            .unsqueeze(1);

    auto resistance_mask = torch::tensor(card_features.resistance_mask, mask_tensor_options).unsqueeze(1);
    auto resistance_tokens = (shared_embedding_holder_->energy_type_embedding_(
                                  torch::tensor(card_features.resistance, index_tensor_options)) *
                              resistance_mask)
                                 .unsqueeze(1);

    auto retreat_cost_mask = torch::tensor(card_features.retreat_cost_mask, mask_tensor_options).unsqueeze(1);
    auto retreat_cost_tokens =
        (retreat_cost_embedding_(torch::tensor(card_features.retreat_cost, index_tensor_options).unsqueeze(-1)) *
         retreat_cost_mask)
            .unsqueeze(1);

    auto number_of_prize_cards_on_knockout_mask =
        torch::tensor(card_features.number_of_prize_cards_on_knockout_mask, mask_tensor_options).unsqueeze(1);
    auto number_of_prize_cards_on_knockout_tokens =
        (number_of_prize_cards_on_knockout_embedding_(
             torch::tensor(card_features.number_of_prize_cards_on_knockout, index_tensor_options).unsqueeze(-1)) *
         number_of_prize_cards_on_knockout_mask)
            .unsqueeze(1);

    auto current_damage_mask = torch::tensor(card_features.current_damage_mask, mask_tensor_options).unsqueeze(1);
    auto current_damage_tokens =
        (current_damage_embedding_(torch::tensor(card_features.current_damage, index_tensor_options).unsqueeze(-1)) *
         current_damage_mask)
            .unsqueeze(1);

    auto [pokemon_turn_trait_tokens, pokemon_turn_trait_mask] =
        embed_flattened_card_feature(pokemon_turn_trait_embedding_, card_features.flattened_pokemon_turn_traits,
                                     card_features.pokemon_turn_trait_card_indices, batch_size);
    auto [provided_energy_tokens, provided_energy_mask] = embed_flattened_card_feature(
        shared_embedding_holder_->energy_type_embedding_, card_features.flattened_provided_energies,
        card_features.provided_energy_card_indices, batch_size);
    auto [attached_energy_tokens, attached_energy_mask] = embed_flattened_card_feature(
        shared_embedding_holder_->energy_type_embedding_, card_features.flattened_attached_energies,
        card_features.attached_energy_card_indices, batch_size);
    return {torch::cat({card_type_tokens, card_subtype_tokens, energy_type_tokens, max_hp_tokens, weakness_tokens,
                        resistance_tokens, retreat_cost_tokens, number_of_prize_cards_on_knockout_tokens,
                        current_damage_tokens, pokemon_turn_trait_tokens, provided_energy_tokens,
                        attached_energy_tokens, instructions_and_conditions_tokens},
                       1),
            torch::cat(
                {card_type_mask, card_subtype_mask, energy_type_mask, max_hp_mask, weakness_mask, resistance_mask,
                 retreat_cost_mask, number_of_prize_cards_on_knockout_mask, current_damage_mask,
                 pokemon_turn_trait_mask, provided_energy_mask, attached_energy_mask, instructions_and_conditions_mask},
                1)};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_flattened_card_feature(
    torch::nn::Embedding& embedding, const std::vector<int64_t>& flattened_card_feature,
    const std::vector<int64_t>& card_indices, int batch_size) {
    auto out_options = torch::TensorOptions().device(device_).dtype(dtype_);
    auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    auto index_options = torch::TensorOptions().device(device_).dtype(torch::kInt64);

    if (flattened_card_feature.empty()) {
        TORCH_CHECK(card_indices.empty(),
                    "embed_flattened_card_feature: card_indices must be empty when flattened_card_feature is empty");
        return {torch::zeros({batch_size, 0, dimension_out_}, out_options),
                torch::zeros({batch_size, 0}, mask_options)};
    }

    TORCH_CHECK(flattened_card_feature.size() == card_indices.size(),
                "embed_flattened_card_feature: flattened_card_feature and card_indices must have the same length");

    auto embedded_feature = embedding(torch::tensor(flattened_card_feature, index_options));

    std::vector<int> next_slot(batch_size, 0);
    const int total_number_of_features = static_cast<int>(card_indices.size());

    std::vector<int64_t> scatter_seq;
    scatter_seq.reserve(static_cast<size_t>(total_number_of_features));
    int max_card_index_repetition = 0;
    for (int card_index : card_indices) {
        const int64_t sequence_index = next_slot[card_index]++;
        scatter_seq.push_back(sequence_index);
        max_card_index_repetition = std::max(max_card_index_repetition, next_slot[card_index]);
    }

    auto out = torch::zeros({batch_size, max_card_index_repetition, dimension_out_}, out_options);
    auto mask = torch::zeros({batch_size, max_card_index_repetition}, mask_options);
    auto scatter_batch_tensor = int64_vector_to_tensor(card_indices, device_);
    auto scatter_seq_tensor = int64_vector_to_tensor(scatter_seq, device_);
    out.index_put_({scatter_batch_tensor, scatter_seq_tensor}, embedded_feature);
    mask.index_put_({scatter_batch_tensor, scatter_seq_tensor}, torch::ones({total_number_of_features}, mask_options));
    return {out, mask};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_instructions_and_conditions(
    const InstructionsAndConditions& instructions_and_conditions, int batch_size) {
    auto embedded_instructions_pair = instruction_embedding_->forward(instructions_and_conditions.instructions);
    auto embedded_conditions_pair = condition_embedding_->forward(instructions_and_conditions.conditions);

    auto [embedded_attacks, mask_attacks] =
        embed_attacks(embedded_instructions_pair, instructions_and_conditions.instruction_attack_indices,
                      instructions_and_conditions.energy_flat, instructions_and_conditions.energy_slot_per_token,
                      instructions_and_conditions.instruction_card_parent_indices, batch_size);
    auto [embedded_abilities, mask_abilities] = embed_ability(
        embedded_instructions_pair, instructions_and_conditions.instruction_ability_indices, embedded_conditions_pair,
        instructions_and_conditions.ability_condition_row_for_instruction_ability,
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
    const std::vector<int64_t>& instruction_attack_indices, const std::vector<int64_t>& energy_flat,
    const std::vector<int64_t>& energy_slot_per_token,
    const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size) {
    if (instruction_attack_indices.empty()) {
        return {torch::zeros({batch_size, 0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_)),
                torch::zeros({batch_size, 0}, torch::TensorOptions().device(device_).dtype(torch::kBool))};
    }

    /// Build attack_energy_sums: Row i matches instruction_attack_indices[i]: sum of energy_type embeddings for that
    /// attack (zeros if no costs).
    const int64_t num_attacks = static_cast<int64_t>(instruction_attack_indices.size());
    auto attack_energy_sums =
        torch::zeros({num_attacks, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
    if (!energy_flat.empty()) {
        auto idx_opts = torch::TensorOptions().device(device_).dtype(torch::kInt64);
        auto energy_idx = torch::tensor(energy_flat, idx_opts);
        auto slot_idx = torch::tensor(energy_slot_per_token, idx_opts);
        auto embedded_energy = shared_embedding_holder_->energy_type_embedding_(energy_idx);
        attack_energy_sums.index_add_(0, slot_idx, embedded_energy);
    }

    const auto attack_instruction_rows = int64_vector_to_tensor(instruction_attack_indices, device_);
    auto embedded_instruction_attacks = embedded_instructions_pair.first.index_select(0, attack_instruction_rows);
    auto embedded_instruction_attacks_mask = embedded_instructions_pair.second.index_select(0, attack_instruction_rows);
    auto embedded_attacks =
        attack_embedding_->forward(attack_energy_sums, embedded_instruction_attacks, embedded_instruction_attacks_mask);
    int max_attacks = 0;
    for (int instruction_attack_index : instruction_attack_indices) {
        int number_of_attacks =
            instruction_card_parent_indices[static_cast<size_t>(instruction_attack_index)].second + 1;
        if (number_of_attacks > max_attacks) max_attacks = number_of_attacks;
    }

    auto out_options = torch::TensorOptions().device(device_).dtype(dtype_);
    auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    auto out = torch::zeros({batch_size, max_attacks, dimension_out_}, out_options);
    auto mask = torch::zeros({batch_size, max_attacks}, mask_options);

    const auto n_attacks = static_cast<int64_t>(instruction_attack_indices.size());
    std::vector<int64_t> scatter_card;
    std::vector<int64_t> scatter_attack_pos;
    scatter_card.reserve(static_cast<size_t>(n_attacks));
    scatter_attack_pos.reserve(static_cast<size_t>(n_attacks));
    for (int64_t row : instruction_attack_indices) {
        const auto& parent = instruction_card_parent_indices[static_cast<size_t>(row)];
        scatter_card.push_back(static_cast<int64_t>(parent.first));
        scatter_attack_pos.push_back(static_cast<int64_t>(parent.second));
    }
    auto scatter_card_t = int64_vector_to_tensor(scatter_card, device_);
    auto scatter_pos_t = int64_vector_to_tensor(scatter_attack_pos, device_);
    out.index_put_({scatter_card_t, scatter_pos_t}, embedded_attacks);
    mask.index_put_({scatter_card_t, scatter_pos_t},
                    torch::ones({n_attacks}, torch::TensorOptions().device(device_).dtype(torch::kBool)));
    return {out, mask};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_ability(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_ability_indices,
    const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
    const std::vector<int64_t>& ability_condition_row_for_instruction_ability,
    const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size) {
    auto out_options = torch::TensorOptions().device(device_).dtype(dtype_);
    auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    if (instruction_ability_indices.empty()) {
        auto empty_slot = torch::zeros({batch_size, 1, dimension_out_}, out_options);
        auto empty_mask = torch::zeros({batch_size, 1}, mask_options);
        return {empty_slot, empty_mask};
    }

    auto instruction_index_tensor = int64_vector_to_tensor(instruction_ability_indices, device_);
    auto embedded_instruction_abilities = embedded_instructions_pair.first.index_select(0, instruction_index_tensor);
    auto embedded_instruction_abilities_mask =
        embedded_instructions_pair.second.index_select(0, instruction_index_tensor);

    const int64_t number_of_abilities = static_cast<int64_t>(instruction_ability_indices.size());
    int64_t max_number_of_conditions = 0;
    for (int64_t condition_index : ability_condition_row_for_instruction_ability) {
        if (condition_index >= 0 && embedded_conditions_pair.first.size(0) > 0) {
            max_number_of_conditions = embedded_conditions_pair.first.size(1);
            break;
        }
    }

    torch::Tensor cond_vals;
    torch::Tensor cond_mask;
    cond_vals = torch::zeros({number_of_abilities, max_number_of_conditions, dimension_out_}, out_options);
    cond_mask = torch::zeros({number_of_abilities, max_number_of_conditions}, mask_options);
    if (max_number_of_conditions > 0) {
        for (int64_t i = 0; i < number_of_abilities; ++i) {
            const int64_t cidx = ability_condition_row_for_instruction_ability[static_cast<size_t>(i)];
            if (cidx >= 0) {
                cond_vals[i].copy_(embedded_conditions_pair.first[cidx]);
                cond_mask[i].copy_(embedded_conditions_pair.second[cidx]);
            }
        }
    }

    auto embedded_abilities = ability_embedding_->forward(embedded_instruction_abilities,
                                                          embedded_instruction_abilities_mask, cond_vals, cond_mask);

    return pad_to_batch(instruction_ability_indices, instruction_card_parent_indices, batch_size, embedded_abilities);
}
std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_instructions(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_card_indices,
    const std::vector<std::pair<int, int>>& instruction_card_parent_indices, int batch_size) {
    if (instruction_card_indices.empty()) {
        auto empty_slot =
            torch::zeros({batch_size, 1, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
        auto empty_mask = torch::zeros({batch_size, 1}, torch::TensorOptions().device(device_).dtype(torch::kBool));
        return {empty_slot, empty_mask};
    }
    auto card_instruction_rows = int64_vector_to_tensor(instruction_card_indices, device_);
    auto embedded_instructions = embedded_instructions_pair.first.index_select(0, card_instruction_rows);
    auto embedded_instructions_mask = embedded_instructions_pair.second.index_select(0, card_instruction_rows);

    auto instruction_query = card_instruction_query_embedding_
                                 ->forward(torch::zeros({embedded_instructions.size(0)},
                                                        torch::TensorOptions().device(device_).dtype(torch::kLong)))
                                 .unsqueeze(1);

    auto pooled_instructions = attention_utils::masked_attention_pooling(
        card_instructions_multi_head_attention_, instruction_query, embedded_instructions, embedded_instructions_mask);

    return pad_to_batch(instruction_card_indices, instruction_card_parent_indices, batch_size, pooled_instructions);
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_conditions(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
    const std::vector<int64_t>& condition_card_indices,
    const std::vector<std::pair<int, int>>& condition_card_parent_indices, int batch_size) {
    if (condition_card_indices.empty()) {
        auto empty_slot =
            torch::zeros({batch_size, 1, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
        auto empty_mask = torch::zeros({batch_size, 1}, torch::TensorOptions().device(device_).dtype(torch::kBool));
        return {empty_slot, empty_mask};
    }
    auto card_condition_rows = int64_vector_to_tensor(condition_card_indices, device_);
    auto embedded_conditions = embedded_conditions_pair.first.index_select(0, card_condition_rows);
    auto embedded_conditions_mask = embedded_conditions_pair.second.index_select(0, card_condition_rows);

    auto condition_query = card_condition_query_embedding_
                               ->forward(torch::zeros({embedded_conditions.size(0)},
                                                      torch::TensorOptions().device(device_).dtype(torch::kLong)))
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

    const auto n = static_cast<int64_t>(card_indices.size());
    std::vector<int64_t> batch_rows;
    batch_rows.reserve(static_cast<size_t>(n));
    for (int64_t idx : card_indices) {
        batch_rows.push_back(static_cast<int64_t>(card_parent_indices[static_cast<size_t>(idx)].first));
    }
    auto batch_rows_t = int64_vector_to_tensor(batch_rows, device_);
    out.index_put_({batch_rows_t}, pooled_tokens);
    mask.index_put_({batch_rows_t}, torch::ones({n}, torch::TensorOptions().device(device_).dtype(torch::kBool)));
    return {out.unsqueeze(1), mask.unsqueeze(1)};
}
