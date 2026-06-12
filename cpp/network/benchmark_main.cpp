#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <torch/cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../network/include/GameEmbedding.h"
#include "../network/include/SharedConstants.h"

namespace serialization = gamecore::serialization;

namespace {

volatile int64_t benchmark_sink = 0;

constexpr serialization::ProtoBufGameInteractionDataType kBenchmarkedGameInteractionDataTypes[] = {
    serialization::GAME_INTERACTION_DATA_TYPE_NUMBER_DATA,
    serialization::GAME_INTERACTION_DATA_TYPE_TARGET_DATA,
    serialization::GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA,
    serialization::GAME_INTERACTION_DATA_TYPE_ATTACK_DATA,
    serialization::GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA,
    serialization::GAME_INTERACTION_DATA_TYPE_ABILITY_DATA,
};

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
        card.add_pokemon_turn_traits(static_cast<serialization::ProtoBufPokemonTurnTrait>((s + i) % 2));
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
            *card.add_instructions() = make_instruction(
                serialization::INSTRUCTION_TYPE_PUT_IN_DECK,
                {make_return_to_deck_type_data(static_cast<int>(seed % 2),
                                               serialization::CARD_POSITION_DISCARD_PILE)});
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

serialization::ProtoBufCard make_card_empty_globals_attack_only() {
    serialization::ProtoBufCard card;
    auto* attack = card.add_attacks();
    attack->add_energy_cost(static_cast<serialization::ProtoBufEnergyType>(1));
    *attack->add_instructions() =
        make_instruction(serialization::INSTRUCTION_TYPE_DEAL_DAMAGE, {make_attack_data(42)});
    apply_card_surface_features(card, 4, 2004);
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

serialization::ProtoBufCardState card_state_from_card(const serialization::ProtoBufCard& card) {
    serialization::ProtoBufCardState state;
    *state.mutable_card() = card;
    return state;
}

void fill_card_states_from_cards(const std::vector<serialization::ProtoBufCard>& cards,
                                 google::protobuf::RepeatedPtrField<serialization::ProtoBufCardState>& out) {
    out.Clear();
    out.Reserve(static_cast<int>(cards.size()));
    for (const auto& c : cards) {
        *out.Add() = card_state_from_card(c);
    }
}

void apply_benchmark_position(serialization::ProtoBufCardState& state, int64_t index) {
    auto* pos = state.mutable_position();
    pos->set_owner(static_cast<serialization::ProtoBufOwner>(index % 2));
    pos->set_opponent_position_knowledge(static_cast<serialization::ProtoBufPositionKnowledge>(index % 3));
    pos->set_top_deck_position_index(static_cast<int32_t>(index % 60));
    pos->clear_possible_positions();
    const int n_pos = 1 + static_cast<int>(index % 4);
    for (int j = 0; j < n_pos; ++j) {
        pos->add_possible_positions(static_cast<serialization::ProtoBufCardPosition>((index + j) % 11));
    }
}

void enrich_card_states_with_positions(google::protobuf::RepeatedPtrField<serialization::ProtoBufCardState>& states) {
    for (int i = 0; i < states.size(); ++i) {
        apply_benchmark_position(*states.Mutable(i), static_cast<int64_t>(i));
    }
}

void assign_deck_ids_to_game_state(serialization::ProtoBufGameState& game_state) {
    for (int i = 0; i < game_state.card_states_size(); ++i) {
        game_state.mutable_card_states(i)->mutable_card()->set_deck_id(i);
    }
}

serialization::ProtoBufPlayerState make_player_state(int64_t seed, bool active, bool attacking,
                                                     int num_turn_traits = 2) {
    serialization::ProtoBufPlayerState player;
    player.set_is_active(active);
    player.set_is_attacking(attacking);
    player.set_knows_his_prizes((seed % 2) == 0);
    player.set_hand_count(static_cast<int32_t>(1 + (seed % 10)));
    player.set_deck_count(static_cast<int32_t>(20 + (seed % 41)));
    player.set_prizes_count(static_cast<int32_t>(seed % 7));
    player.set_bench_count(static_cast<int32_t>(seed % 6));
    player.set_discard_pile_count(static_cast<int32_t>((seed * 3) % 50));
    player.set_turn_counter(static_cast<int32_t>(1 + (seed % 10)));
    for (int i = 0; i < num_turn_traits; ++i) {
        player.add_player_turn_traits(static_cast<serialization::ProtoBufPlayerTurnTrait>((seed + i) % 4));
    }
    return player;
}

serialization::ProtoBufGameState make_game_state(int64_t card_count, int64_t seed, int self_turn_traits = 2,
                                                 int opponent_turn_traits = 2) {
    serialization::ProtoBufGameState game_state;
    game_state.set_recreatable(true);
    game_state.set_technical_game_state(serialization::GAME_STATE_IDLE_PLAYER_TURN);
    *game_state.mutable_self_state() = make_player_state(seed, true, true, self_turn_traits);
    *game_state.mutable_opponent_state() = make_player_state(seed + 1, false, false, opponent_turn_traits);

    google::protobuf::RepeatedPtrField<serialization::ProtoBufCardState> states;
    fill_card_states_from_cards(build_card_batch(card_count), states);
    enrich_card_states_with_positions(states);
    for (const auto& state : states) {
        *game_state.add_card_states() = state;
    }
    assign_deck_ids_to_game_state(game_state);
    return game_state;
}

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

torch::Tensor extract_card_embeddings(const torch::Tensor& game_state_embedding) {
    return game_state_embedding.slice(1, 2, game_state_embedding.size(1));
}

const char* game_interaction_data_type_name(serialization::ProtoBufGameInteractionDataType type) {
    switch (type) {
        case serialization::GAME_INTERACTION_DATA_TYPE_NUMBER_DATA:
            return "NUMBER_DATA";
        case serialization::GAME_INTERACTION_DATA_TYPE_TARGET_DATA:
            return "TARGET_DATA";
        case serialization::GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA:
            return "INTERACTION_CARD_DATA";
        case serialization::GAME_INTERACTION_DATA_TYPE_ATTACK_DATA:
            return "ATTACK_DATA";
        case serialization::GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA:
            return "SELECT_FROM_DATA";
        case serialization::GAME_INTERACTION_DATA_TYPE_ABILITY_DATA:
            return "ABILITY_DATA";
        default:
            return "UNKNOWN";
    }
}

serialization::ProtoBufGameInteractionData make_game_interaction_data_for_type(
    serialization::ProtoBufGameInteractionDataType type, int64_t seed) {
    serialization::ProtoBufGameInteractionData data;
    data.set_data_type(type);
    switch (type) {
        case serialization::GAME_INTERACTION_DATA_TYPE_NUMBER_DATA:
            data.mutable_number_data()->set_number(1 + static_cast<int32_t>(seed % 5));
            break;
        case serialization::GAME_INTERACTION_DATA_TYPE_TARGET_DATA: {
            auto* target = data.mutable_target_data();
            target->add_possible_targets(0);
            target->add_possible_targets(1);
            target->add_possible_targets(2);
            target->set_target_action(serialization::ACTION_ON_SELECTION_DISCARD);
            target->set_remainder_action(serialization::ACTION_ON_SELECTION_TAKE_TO_HAND);
            target->set_allow_multiple_times(seed % 2 == 0);
            target->set_number_of_targets(2);
            break;
        }
        case serialization::GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA:
            data.mutable_interaction_card_data()->set_card(static_cast<int32_t>(seed % 3));
            break;
        case serialization::GAME_INTERACTION_DATA_TYPE_ATTACK_DATA: {
            const auto attack_card = make_card_empty_globals_attack_only();
            *data.mutable_attack_data()->mutable_attack() = attack_card.attacks(0);
            break;
        }
        case serialization::GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA:
            data.mutable_select_from_data()->set_select_from(
                static_cast<serialization::ProtoBufSelectFrom>((seed % 2) == 0
                                                                 ? serialization::SELECT_FROM_DISCARD_PILE
                                                                 : serialization::SELECT_FROM_DECK));
            break;
        case serialization::GAME_INTERACTION_DATA_TYPE_ABILITY_DATA: {
            const auto ability_card = make_card_for_variant(3, seed);
            *data.mutable_ability_data()->mutable_ability() = ability_card.ability();
            break;
        }
        default:
            break;
    }
    return data;
}

int64_t seed_offset_for_type(serialization::ProtoBufGameInteractionDataType type) {
    switch (type) {
        case serialization::GAME_INTERACTION_DATA_TYPE_NUMBER_DATA:
            return 0;
        case serialization::GAME_INTERACTION_DATA_TYPE_TARGET_DATA:
            return 100;
        case serialization::GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA:
            return 200;
        case serialization::GAME_INTERACTION_DATA_TYPE_ATTACK_DATA:
            return 300;
        case serialization::GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA:
            return 400;
        case serialization::GAME_INTERACTION_DATA_TYPE_ABILITY_DATA:
            return 500;
        default:
            return 0;
    }
}

std::vector<serialization::ProtoBufGameInteraction> make_game_interaction_batch(
    int64_t batch_size, std::optional<serialization::ProtoBufGameInteractionDataType> excluded_type) {
    std::vector<serialization::ProtoBufGameInteraction> batch;
    batch.reserve(static_cast<size_t>(batch_size));
    for (int64_t i = 0; i < batch_size; ++i) {
        serialization::ProtoBufGameInteraction interaction;
        interaction.set_type(serialization::GAME_INTERACTION_TYPE_SELECT_CARDS);
        for (const auto type : kBenchmarkedGameInteractionDataTypes) {
            if (excluded_type.has_value() && excluded_type.value() == type) {
                continue;
            }
            *interaction.add_data() = make_game_interaction_data_for_type(type, i + seed_offset_for_type(type));
        }
        batch.push_back(std::move(interaction));
    }
    return batch;
}

struct GameInteractionBenchCase {
    std::string name;
    serialization::ProtoBufGameState game_state;
    torch::Tensor card_indices;
    torch::Tensor cards;
    std::vector<serialization::ProtoBufGameInteraction> interactions;
};

GameInteractionBenchCase build_game_interaction_bench_case(GameEmbeddingImpl& game_embedding, const torch::Device& device,
                                                           const std::string& name, int64_t card_count, int64_t seed,
                                                           int64_t interaction_batch_size,
                                                           std::optional<serialization::ProtoBufGameInteractionDataType>
                                                               excluded_type) {
    GameInteractionBenchCase bench_case;
    bench_case.name = name;
    bench_case.game_state = make_game_state(card_count, seed);
    const auto [game_state_embedding, mask, card_indices] =
        game_embedding.embedGameState({bench_case.game_state});
    (void)mask;
    bench_case.card_indices = card_indices;
    bench_case.cards = extract_card_embeddings(game_state_embedding);
    bench_case.interactions = make_game_interaction_batch(interaction_batch_size, excluded_type);
    return bench_case;
}

std::vector<GameInteractionBenchCase> build_game_interaction_bench_cases(GameEmbeddingImpl& game_embedding,
                                                                         const torch::Device& device,
                                                                         int64_t card_count, int64_t seed,
                                                                         int64_t interaction_batch_size) {
    std::vector<GameInteractionBenchCase> cases;
    cases.push_back(build_game_interaction_bench_case(game_embedding, device, "all_types", card_count, seed,
                                                      interaction_batch_size, std::nullopt));
    for (const auto excluded_type : kBenchmarkedGameInteractionDataTypes) {
        const std::string name = std::string("without_") + game_interaction_data_type_name(excluded_type);
        cases.push_back(build_game_interaction_bench_case(game_embedding, device, name, card_count, seed + 1000,
                                                          interaction_batch_size, excluded_type));
    }
    return cases;
}

void verify_game_embedding_output_shape(GameEmbeddingImpl& game_embedding, const torch::Device& device,
                                        int64_t dimension_out, const std::string& label) {
    for (int64_t card_count : {0, 32, 128}) {
        auto game_state = make_game_state(card_count, card_count + 10);
        auto [out, mask, card_indices] = game_embedding.embedGameState({game_state});
        (void)card_indices;
        const int64_t expected_rows = card_count + 2;
        if (out.dim() != 3 || out.size(0) != 1 || out.size(1) != expected_rows || out.size(2) != dimension_out) {
            std::cerr << label << " GameEmbedding::embedGameState shape check failed for " << card_count
                      << " cards: expected (1, " << expected_rows << ", " << dimension_out << ")\n";
            std::abort();
        }
        if (mask.dim() != 2 || mask.size(0) != 1 || mask.size(1) != expected_rows) {
            std::cerr << label << " GameEmbedding::embedGameState mask shape check failed for " << card_count << "\n";
            std::abort();
        }
        if (!tensor_device_matches_module(out, device)) {
            std::cerr << label << " GameEmbedding::embedGameState device mismatch\n";
            std::abort();
        }
        benchmark_sink += out.numel();
    }
    {
        auto uneven = make_game_state(0, 500, 0, 4);
        auto [out, mask, card_indices] = game_embedding.embedGameState({uneven});
        (void)card_indices;
        (void)mask;
        if (out.dim() != 3 || out.size(0) != 1 || out.size(1) != 2 || out.size(2) != dimension_out) {
            std::cerr << label << " GameEmbedding::embedGameState uneven traits (0 cards): expected (1, 2, "
                      << dimension_out << ")\n";
            std::abort();
        }
        if (!tensor_device_matches_module(out, device)) {
            std::cerr << label << " GameEmbedding::embedGameState uneven traits device mismatch\n";
            std::abort();
        }
        benchmark_sink += out.numel();
    }

    {
        auto [empty_embedding, empty_mask, empty_indices] = game_embedding.embedGameState({make_game_state(0, 600)});
        (void)empty_embedding;
        (void)empty_mask;
        auto empty_cards =
            torch::empty({1, 0, dimension_out}, torch::TensorOptions().device(device).dtype(torch::kFloat));
        auto [out, interaction_mask] = game_embedding.embedGameInteraction({{}}, empty_indices, empty_cards);
        (void)interaction_mask;
        if (out.dim() != 3 || out.size(0) != 1 || out.size(1) != 0 || out.size(2) != dimension_out) {
            std::cerr << label << " GameEmbedding::embedGameInteraction empty batch shape check failed: expected (1, 0, "
                      << dimension_out << ")\n";
            std::abort();
        }
        if (!tensor_device_matches_module(out, device)) {
            std::cerr << label << " GameEmbedding::embedGameInteraction empty batch device mismatch\n";
            std::abort();
        }
        benchmark_sink += out.numel();
    }

    for (const int64_t interaction_batch_size : {32, 128}) {
        const auto cases = build_game_interaction_bench_cases(game_embedding, device, interaction_batch_size,
                                                              700 + interaction_batch_size, interaction_batch_size);
        for (const auto& bench_case : cases) {
            auto [out, interaction_mask] = game_embedding.embedGameInteraction({bench_case.interactions},
                                                                             bench_case.card_indices, bench_case.cards);
            (void)interaction_mask;
            const int64_t expected_rows = static_cast<int64_t>(bench_case.interactions.size());
            if (out.dim() != 3 || out.size(0) != 1 || out.size(1) != expected_rows || out.size(2) != dimension_out) {
                std::cerr << label << " GameEmbedding::embedGameInteraction shape check failed [" << bench_case.name
                          << ", batch=" << interaction_batch_size << "]: expected (1, " << expected_rows << ", "
                          << dimension_out << ")\n";
                std::abort();
            }
            if (!tensor_device_matches_module(out, device)) {
                std::cerr << label << " GameEmbedding::embedGameInteraction device mismatch [" << bench_case.name
                          << "]\n";
                std::abort();
            }
            benchmark_sink += out.numel();
        }
    }

    std::cout << label << " GameEmbedding shape checks passed\n";
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

    auto game_embedding = std::make_shared<GameEmbeddingImpl>(dimension, device, dtype);
    game_embedding->eval();

    verify_game_embedding_output_shape(*game_embedding, device, dimension, label);

    std::cout << "\n== " << label << " ==\n";

    auto game_state_32 = make_game_state(32, 200);
    benchmark_ms(label + " game_embedding_embed_game_state_32", device, warmup_runs, measured_runs, [&]() {
        auto [out, mask, card_indices] = game_embedding->embedGameState({game_state_32});
        (void)mask;
        (void)card_indices;
        benchmark_sink += out.numel();
    });

    auto game_state_128 = make_game_state(128, 300);
    benchmark_ms(label + " game_embedding_embed_game_state_128", device, warmup_runs, measured_runs, [&]() {
        auto [out, mask, card_indices] = game_embedding->embedGameState({game_state_128});
        (void)mask;
        (void)card_indices;
        benchmark_sink += out.numel();
    });

    auto game_state_32_uneven = make_game_state(32, 400, 0, 4);
    benchmark_ms(label + " game_embedding_embed_game_state_32_uneven_traits", device, warmup_runs, measured_runs,
                 [&]() {
                     auto [out, mask, card_indices] = game_embedding->embedGameState({game_state_32_uneven});
                     (void)mask;
                     (void)card_indices;
                     benchmark_sink += out.numel();
                 });

    auto game_state_128_uneven = make_game_state(128, 500, 4, 0);
    benchmark_ms(label + " game_embedding_embed_game_state_128_uneven_traits", device, warmup_runs, measured_runs,
                 [&]() {
                     auto [out, mask, card_indices] = game_embedding->embedGameState({game_state_128_uneven});
                     (void)mask;
                     (void)card_indices;
                     benchmark_sink += out.numel();
                 });

    for (const int64_t interaction_batch_size : {32, 128}) {
        const auto interaction_cases = build_game_interaction_bench_cases(
            *game_embedding, device, interaction_batch_size, 800 + interaction_batch_size, interaction_batch_size);
        for (const auto& bench_case : interaction_cases) {
            const std::string bench_name = label + " game_embedding_embed_game_interaction_" +
                                           std::to_string(interaction_batch_size) + "_" + bench_case.name;
            benchmark_ms(bench_name, device, warmup_runs, measured_runs, [&]() {
                auto [out, interaction_mask] = game_embedding->embedGameInteraction({bench_case.interactions},
                                                                                    bench_case.card_indices,
                                                                                    bench_case.cards);
                (void)interaction_mask;
                benchmark_sink += out.numel();
            });
        }
    }
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
