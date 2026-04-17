#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <torch/cuda.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../network/include/CardEmbedding.h"
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

/// Deterministic scalar and repeated fields consumed by `CardEmbedding::collect_card_features`.
/// Optional proto fields are only set when their bitmask slot is active so mask paths stay covered.
void apply_card_surface_features(serialization::ProtoBufCard& card, int variant, int64_t seed) {
    const int v = variant % 12;
    const int s = static_cast<int>(seed);

    card.set_card_type(static_cast<serialization::ProtoBufCardType>(1 + (s % 3)));
    card.set_card_subtype(static_cast<serialization::ProtoBufCardSubtype>(1 + ((s + v) % 9)));

    const int opt_mask = (v * 17 + s) & 0x3f;
    if (opt_mask & 1) {
        card.set_energy_type(static_cast<serialization::ProtoBufEnergyType>(1 + (s % 10)));
    }
    if (opt_mask & 2) {
        card.set_max_hp(30 + (s % 300));
    }
    if (opt_mask & 4) {
        card.set_weakness(static_cast<serialization::ProtoBufEnergyType>(1 + ((s + 1) % 10)));
    }
    if (opt_mask & 8) {
        card.set_resistance(static_cast<serialization::ProtoBufEnergyType>(1 + ((s + 2) % 10)));
    }
    if (opt_mask & 16) {
        card.set_retreat_cost(1 + (s % 4));
    }
    if (opt_mask & 32) {
        card.set_number_of_prize_cards_on_knockout(1 + (s % 3));
    }
    if (((v + s) % 5) != 0) {
        card.set_current_damage(s % 200);
    }

    const int n_traits = 1 + (v % 2);
    for (int i = 0; i < n_traits; ++i) {
        card.add_pokemon_turn_traits(
            static_cast<serialization::ProtoBufPokemonTurnTrait>((s + i) % 2));
    }
    const int n_provided = 2 + (v % 3);
    for (int i = 0; i < n_provided; ++i) {
        card.add_provided_energy(static_cast<serialization::ProtoBufEnergyType>(1 + ((s + i) % 10)));
    }
    const int n_attached = 3 + (v % 2);
    for (int i = 0; i < n_attached; ++i) {
        card.add_attached_energy(static_cast<serialization::ProtoBufEnergyType>(1 + ((s * 3 + i) % 10)));
    }
}

/// All optional card-level scalars and repeated energies/traits set; no instructions, conditions, ability, or attacks.
void apply_all_optional_card_fields(serialization::ProtoBufCard& card, int64_t seed) {
    const int s = static_cast<int>(seed);
    card.set_card_type(static_cast<serialization::ProtoBufCardType>(1 + (s % 3)));
    card.set_card_subtype(static_cast<serialization::ProtoBufCardSubtype>(1 + (s % 9)));
    card.set_energy_type(static_cast<serialization::ProtoBufEnergyType>(1 + (s % 10)));
    card.set_max_hp(30 + (s % 300));
    card.set_weakness(static_cast<serialization::ProtoBufEnergyType>(1 + ((s + 1) % 10)));
    card.set_resistance(static_cast<serialization::ProtoBufEnergyType>(1 + ((s + 2) % 10)));
    card.set_retreat_cost(1 + (s % 4));
    card.set_number_of_prize_cards_on_knockout(1 + (s % 3));
    card.set_current_damage(s % 200);
    card.add_pokemon_turn_traits(serialization::POKEMON_TURN_TRAIT_PUT_IN_PLAY_THIS_TURN);
    card.add_pokemon_turn_traits(serialization::POKEMON_TURN_TRAIT_ABILITY_USED_THIS_TURN);
    for (int i = 0; i < 4; ++i) {
        card.add_provided_energy(static_cast<serialization::ProtoBufEnergyType>(1 + (i % 10)));
        card.add_attached_energy(static_cast<serialization::ProtoBufEnergyType>(1 + ((i + 5) % 10)));
    }
}

serialization::ProtoBufCard make_card_high_repeat_lists() {
    serialization::ProtoBufCard card;
    card.set_card_type(serialization::CARD_TYPE_POKEMON);
    card.set_card_subtype(serialization::CARD_SUBTYPE_BASIC_POKEMON);
    card.set_max_hp(90);
    card.set_energy_type(serialization::ENERGY_TYPE_WATER);
    for (int i = 0; i < 8; ++i) {
        card.add_pokemon_turn_traits(
            static_cast<serialization::ProtoBufPokemonTurnTrait>(i % 2));
    }
    for (int i = 0; i < 12; ++i) {
        card.add_attached_energy(static_cast<serialization::ProtoBufEnergyType>(1 + (i % 10)));
    }
    for (int i = 0; i < 6; ++i) {
        card.add_provided_energy(static_cast<serialization::ProtoBufEnergyType>(1 + (i % 10)));
    }
    return card;
}

serialization::ProtoBufCard make_card_all_optionals_static_only() {
    serialization::ProtoBufCard card;
    apply_all_optional_card_fields(card, 1001);
    return card;
}

/// One ability per card max. Twelve distinct shapes for coverage (indexed by `variant % 12`).
/// Several variants omit card-level instructions and/or conditions while still using ability- or attack-level
/// content. Empty global instruction or condition lists are supported by `CardEmbedding::forward`.
serialization::ProtoBufCard make_card_for_variant(int variant, int64_t seed) {
    serialization::ProtoBufCard card;
    switch (variant % 12) {
        case 0:
            *card.add_instructions() = make_instruction(serialization::INSTRUCTION_TYPE_SHOW_CARDS, {});
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            break;
        case 1:
            *card.add_instructions() = make_instruction(
                serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                {make_attack_data(7 + static_cast<int>(seed % 50))});
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            break;
        case 2:
            *card.add_instructions() = make_instruction(
                serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                {make_attack_data(10 + static_cast<int>(seed % 40))});
            *card.add_conditions() =
                make_condition(serialization::CONDITION_TYPE_HAS_CARDS,
                               {make_card_amount_data(1, 4, serialization::CARD_POSITION_DECK),
                                make_filter_data(make_nested_filter(seed, 0))});
            break;
        case 3: {
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            auto* ability = card.mutable_ability();
            *ability->add_instructions() =
                make_instruction(serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                                 {make_attack_data(15 + static_cast<int>(seed % 30))});
            break;
        }
        case 4: {
            auto* ability = card.mutable_ability();
            *ability->add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            *ability->add_instructions() =
                make_instruction(serialization::INSTRUCTION_TYPE_DISCARD, {make_discard_data(0)});
            auto* attack = card.add_attacks();
            attack->add_energy_cost(static_cast<serialization::ProtoBufEnergyType>(1));
            *attack->add_instructions() =
                make_instruction(serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                                 {make_attack_data(20 + static_cast<int>(seed % 25))});
            break;
        }
        case 5:
            *card.add_instructions() = make_instruction(serialization::INSTRUCTION_TYPE_SHOW_CARDS, {});
            *card.add_conditions() =
                make_condition(serialization::CONDITION_TYPE_HAS_CARDS,
                               {make_card_amount_data(1, 8, serialization::CARD_POSITION_HAND),
                                make_filter_data(make_leaf_filter(serialization::FILTER_TYPE_CARD_TYPE,
                                                                  serialization::FILTER_OPERATION_EQUALS, 1))});
            break;
        case 6: {
            auto* ability = card.mutable_ability();
            *ability->add_conditions() =
                make_condition(serialization::CONDITION_TYPE_HAS_CARDS,
                               {make_card_amount_data(1, 3, serialization::CARD_POSITION_DECK),
                                make_filter_data(make_nested_filter(seed, 1))});
            *ability->add_instructions() = make_instruction(serialization::INSTRUCTION_TYPE_SHOW_CARDS, {});
            break;
        }
        case 7:
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            for (int a = 0; a < 2; ++a) {
                auto* attack = card.add_attacks();
                attack->add_energy_cost(static_cast<serialization::ProtoBufEnergyType>((seed + a) % 3 + 1));
                *attack->add_instructions() =
                    make_instruction(serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                                     {make_attack_data(12 + a + static_cast<int>(seed % 20))});
            }
            break;
        case 8:
            *card.add_instructions() = make_instruction(
                serialization::INSTRUCTION_TYPE_SELECT_CARDS,
                {make_card_amount_data(1, 2, serialization::CARD_POSITION_HAND),
                 make_filter_data(make_nested_filter(seed, 2))});
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            {
                auto* ability = card.mutable_ability();
                *ability->add_instructions() =
                    make_instruction(serialization::INSTRUCTION_TYPE_SHUFFLE_DECK,
                                     {make_player_target_data(static_cast<int>(seed % 2))});
            }
            break;
        case 9:
            *card.add_instructions() = make_instruction(serialization::INSTRUCTION_TYPE_PUT_IN_DECK,
                                                          {make_return_to_deck_type_data(
                                                              static_cast<int>(seed % 2), serialization::CARD_POSITION_DISCARD_PILE)});
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            {
                auto* ability = card.mutable_ability();
                *ability->add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
                *ability->add_instructions() =
                    make_instruction(serialization::INSTRUCTION_TYPE_REVEAL_CARDS,
                                     {make_card_amount_data(1, 2, serialization::CARD_POSITION_SELECTED_CARDS),
                                      make_filter_data(make_leaf_filter(serialization::FILTER_TYPE_EXCLUDE_SOURCE,
                                                                        serialization::FILTER_OPERATION_NONE, 0))});
            }
            break;
        case 10:
            *card.add_instructions() = make_instruction(
                serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                {make_attack_data(5 + static_cast<int>(seed % 50))});
            *card.add_instructions() = make_instruction(
                serialization::INSTRUCTION_TYPE_DISCARD,
                {make_discard_data(static_cast<int>(seed % 3))});
            *card.add_instructions() = make_instruction(
                serialization::INSTRUCTION_TYPE_TAKE_TO_HAND,
                {make_card_amount_data(1, 1, serialization::CARD_POSITION_DECK)});
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            break;
        case 11:
            *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
            {
                auto* attack = card.add_attacks();
                *attack->add_instructions() =
                    make_instruction(serialization::INSTRUCTION_TYPE_DEAL_DAMAGE,
                                     {make_attack_data(30 + static_cast<int>(seed % 40))});
            }
            break;
        default:
            break;
    }
    apply_card_surface_features(card, variant % 12, seed);
    return card;
}

std::vector<serialization::ProtoBufCard> build_card_batch(int64_t batch_size) {
    std::vector<serialization::ProtoBufCard> cards;
    cards.reserve(static_cast<size_t>(batch_size));
    for (int64_t i = 0; i < batch_size; ++i) {
        cards.push_back(make_card_for_variant(static_cast<int>(i % 12), i));
    }
    return cards;
}

serialization::ProtoBufCard make_card_empty_global_instructions_one_condition() {
    serialization::ProtoBufCard card;
    *card.add_conditions() = make_condition(serialization::CONDITION_TYPE_ABILITY_NOT_USED, {});
    apply_card_surface_features(card, 0, 2002);
    return card;
}

serialization::ProtoBufCard make_card_empty_global_conditions_one_instruction() {
    serialization::ProtoBufCard card;
    *card.add_instructions() = make_instruction(serialization::INSTRUCTION_TYPE_SHOW_CARDS, {});
    apply_card_surface_features(card, 1, 2003);
    return card;
}

serialization::ProtoBufCard make_card_empty_globals_attack_only() {
    serialization::ProtoBufCard card;
    auto* attack = card.add_attacks();
    attack->add_energy_cost(static_cast<serialization::ProtoBufEnergyType>(1));
    *attack->add_instructions() =
        make_instruction(serialization::INSTRUCTION_TYPE_DEAL_DAMAGE, {make_attack_data(42)});
    apply_card_surface_features(card, 4, 2004);
    return card;
}

serialization::ProtoBufCard make_card_completely_empty() { return serialization::ProtoBufCard{}; }

/// `torch::Device(kCUDA)` uses index -1 (unspecified); tensors use an explicit index (e.g. 0). Treat -1 as 0.
bool tensor_device_matches_module(const torch::Tensor& tensor, const torch::Device& module_device) {
    const torch::Device& tdev = tensor.device();
    if (tdev.type() != module_device.type()) {
        return false;
    }
    if (tdev.is_cuda()) {
        const int ti = tdev.index() < 0 ? 0 : tdev.index();
        const int mi = module_device.index() < 0 ? 0 : module_device.index();
        return ti == mi;
    }
    return true;
}

void verify_card_embedding_output_shape(CardEmbeddingImpl& card_embedding, const torch::Device& device,
                                        int64_t dimension_out, const std::string& label) {
    auto check = [&](const std::vector<serialization::ProtoBufCard>& cards, const char* case_name) {
        auto [out, adj] = card_embedding.forward(cards);
        const int64_t n = static_cast<int64_t>(cards.size());
        if (out.dim() != 2 || out.size(0) != n || out.size(1) != dimension_out) {
            std::cerr << label << " CardEmbedding::forward shape check failed [" << case_name << "]: expected (" << n
                      << ", " << dimension_out << "), got (";
            for (int64_t d = 0; d < out.dim(); ++d) {
                std::cerr << out.size(d) << (d + 1 < out.dim() ? ", " : "");
            }
            std::cerr << ")\n";
            std::abort();
        }
        if (!tensor_device_matches_module(out, device)) {
            std::cerr << label << " CardEmbedding::forward device mismatch [" << case_name << "]: output ("
                      << static_cast<int>(out.device().type()) << "," << out.device().index() << ") module ("
                      << static_cast<int>(device.type()) << "," << device.index() << ")\n";
            std::abort();
        }
        const auto& sparse = adj.evolves_from_adjacency;
        if (!sparse.is_sparse()) {
            std::cerr << label << " CardEmbedding::forward evolves_from_adjacency must be sparse COO [" << case_name
                      << "]\n";
            std::abort();
        }
        if (sparse.size(0) != n || sparse.size(1) != n) {
            std::cerr << label << " CardEmbedding::forward sparse adjacency shape failed [" << case_name
                      << "]: expected (" << n << ", " << n << ")\n";
            std::abort();
        }
        if (!tensor_device_matches_module(sparse, device)) {
            std::cerr << label << " CardEmbedding::forward sparse adjacency device mismatch [" << case_name << "]\n";
            std::abort();
        }
        const auto& attached_sparse = adj.attached_energy_adjacency;
        if (!attached_sparse.is_sparse()) {
            std::cerr << label << " CardEmbedding::forward attached_energy_adjacency must be sparse COO [" << case_name
                      << "]\n";
            std::abort();
        }
        if (attached_sparse.size(0) != n || attached_sparse.size(1) != n) {
            std::cerr << label << " CardEmbedding::forward attached_energy sparse shape failed [" << case_name
                      << "]: expected (" << n << ", " << n << ")\n";
            std::abort();
        }
        if (!tensor_device_matches_module(attached_sparse, device)) {
            std::cerr << label << " CardEmbedding::forward attached_energy_adjacency device mismatch [" << case_name
                      << "]\n";
            std::abort();
        }
        benchmark_sink +=
            out.numel() + sparse.values().size(0) + attached_sparse.values().size(0);
    };

    for (int v = 0; v < 12; ++v) {
        const std::string tag = "single_variant_" + std::to_string(v);
        check({make_card_for_variant(v, static_cast<int64_t>(v))}, tag.c_str());
    }

    std::vector<serialization::ProtoBufCard> one_of_each;
    one_of_each.reserve(12);
    for (int v = 0; v < 12; ++v) {
        one_of_each.push_back(make_card_for_variant(v, static_cast<int64_t>(v)));
    }
    check(one_of_each, "batch_twelve_variants");

    check(build_card_batch(1), "batch_1_mixed");
    check(build_card_batch(7), "batch_7_mixed");
    check(build_card_batch(64), "batch_64_mixed");
    check(build_card_batch(static_cast<int64_t>(256)), "batch_256_mixed");

    check({make_card_empty_global_instructions_one_condition()}, "empty_global_instructions_one_condition");
    check({make_card_empty_global_conditions_one_instruction()}, "empty_global_conditions_one_instruction");
    check({make_card_empty_globals_attack_only()}, "empty_globals_attack_only");
    check({make_card_completely_empty()}, "completely_empty_card");
    check({make_card_empty_global_instructions_one_condition(), make_card_empty_global_conditions_one_instruction(),
           make_card_empty_globals_attack_only(), make_card_completely_empty()},
          "batch_mixed_empty_global_and_empty_card");

    check({make_card_high_repeat_lists()}, "high_repeat_traits_and_energies");
    check({make_card_all_optionals_static_only()}, "all_optionals_static_only");
    check({make_card_high_repeat_lists(), make_card_all_optionals_static_only()}, "batch_high_repeat_and_static_only");

    std::cout << label << " CardEmbedding::forward shape checks passed (expected [num_cards, " << dimension_out
              << "])\n";
}

void synchronize_device(const torch::Device& device) {
    if (device.is_cuda()) {
        const int idx = device.index() < 0 ? 0 : device.index();
        torch::cuda::synchronize(idx);
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
    auto card_embedding = std::make_shared<CardEmbeddingImpl>(dimension, device, dtype);

    shared->eval();
    instruction_data_embedding->eval();
    instruction_embedding->eval();
    condition_embedding->eval();
    card_embedding->eval();

    verify_card_embedding_output_shape(*card_embedding, device, dimension, label);

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

    auto cards_large = build_card_batch(instruction_batch_size);
    benchmark_ms(label + " card_embedding_forward_256", device, warmup_runs, measured_runs, [&]() {
        auto [out, adj] = card_embedding->forward(cards_large);
        benchmark_sink += out.numel() + adj.evolves_from_adjacency.values().size(0) +
                          adj.attached_energy_adjacency.values().size(0);
    });

    auto cards_32 = build_card_batch(32);
    benchmark_ms(label + " card_embedding_forward_32", device, warmup_runs, measured_runs, [&]() {
        auto [out, adj] = card_embedding->forward(cards_32);
        benchmark_sink += out.numel() + adj.evolves_from_adjacency.values().size(0) +
                          adj.attached_energy_adjacency.values().size(0);
    });

    auto cards_128 = build_card_batch(128);
    benchmark_ms(label + " card_embedding_forward_128", device, warmup_runs, measured_runs, [&]() {
        auto [out, adj] = card_embedding->forward(cards_128);
        benchmark_sink += out.numel() + adj.evolves_from_adjacency.values().size(0) +
                          adj.attached_energy_adjacency.values().size(0);
    });

    std::vector<serialization::ProtoBufCard> cards_homogeneous;
    cards_homogeneous.reserve(static_cast<size_t>(instruction_batch_size));
    for (int64_t i = 0; i < instruction_batch_size; ++i) {
        cards_homogeneous.push_back(make_card_for_variant(4, i));
    }
    benchmark_ms(label + " card_embedding_forward_256_all_variant4", device, warmup_runs, measured_runs, [&]() {
        auto [out, adj] = card_embedding->forward(cards_homogeneous);
        benchmark_sink += out.numel() + adj.evolves_from_adjacency.values().size(0) +
                          adj.attached_energy_adjacency.values().size(0);
    });
}

}  // namespace

int main() {
    torch::InferenceMode guard;
    run_embedding_benchmarks(torch::Device(torch::kCPU), "cpu");
    if (torch::cuda::is_available()) {
        const torch::Device cuda_dev(torch::kCUDA, 0);
        run_embedding_benchmarks(cuda_dev, "cuda");
        with_deterministic_algorithms(true, [&]() { run_embedding_benchmarks(cuda_dev, "cuda_deterministic"); });
    } else {
        std::cout << "\nCUDA benchmark skipped: CUDA is not available.\n";
    }

    std::cout << "Benchmark sink: " << benchmark_sink << "\n";
    return 0;
}
