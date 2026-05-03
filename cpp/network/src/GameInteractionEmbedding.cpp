#include "network/include/GameInteractionEmbedding.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include <cstdint>

#include "network/include/SharedConstants.h"

using ProtoBufGameInteractionData = gamecore::serialization::ProtoBufGameInteractionData;
using ProtoBufGameInteractionDataType = gamecore::serialization::ProtoBufGameInteractionDataType;
using ProtoBufConditionalTargetQuery = gamecore::serialization::ProtoBufConditionalTargetQuery;
using ProtoBufAbility = gamecore::serialization::ProtoBufAbility;
namespace {

template <typename T>
void append_repeated_proto_to_vector(const google::protobuf::RepeatedPtrField<T>& src, std::vector<T>& dst) {
    dst.insert(dst.end(), src.begin(), src.end());
}

void append_attack_instructions_and_conditions(const ProtoBufAttack& attack,
                                               InstructionsAndConditions& instructions_and_conditions,
                                               int64_t batch_index) {
    if (attack.instructions_size() > 0) {
        if (attack.energy_cost_size() > 0) {
            for (const auto energy_type : attack.energy_cost()) {
                instructions_and_conditions.energy_flat.push_back(static_cast<int64_t>(energy_type));
                instructions_and_conditions.energy_slot_per_token.push_back(0);  // only one attack per game interaction
            }
        }
        instructions_and_conditions.instructions.emplace_back();
        append_repeated_proto_to_vector<ProtoBufInstruction>(attack.instructions(),
                                                             instructions_and_conditions.instructions.back());
        instructions_and_conditions.instruction_card_parent_indices.push_back({static_cast<int>(batch_index), 0});
        instructions_and_conditions.instruction_card_indices.push_back(
            static_cast<int64_t>(instructions_and_conditions.instructions.size() - 1));
    }
}

void append_ability_instructions_and_conditions(const ProtoBufAbility& ability,
                                                InstructionsAndConditions& instructions_and_conditions,
                                                int64_t batch_index) {
    int64_t ability_condition_row = -1;
    if (ability.conditions_size() > 0) {
        instructions_and_conditions.conditions.emplace_back();
        append_repeated_proto_to_vector<ProtoBufCondition>(ability.conditions(),
                                                           instructions_and_conditions.conditions.back());
        instructions_and_conditions.condition_card_parent_indices.push_back({static_cast<int>(batch_index), 0});
        ability_condition_row = static_cast<int64_t>(instructions_and_conditions.conditions.size() - 1);
    }
    if (ability.instructions_size() > 0) {
        instructions_and_conditions.instructions.emplace_back();
        append_repeated_proto_to_vector<ProtoBufInstruction>(ability.instructions(),
                                                             instructions_and_conditions.instructions.back());
        instructions_and_conditions.instruction_card_parent_indices.push_back({static_cast<int>(batch_index), 0});
        instructions_and_conditions.instruction_ability_indices.push_back(
            static_cast<int64_t>(instructions_and_conditions.instructions.size() - 1));
        instructions_and_conditions.ability_condition_row_for_instruction_ability.push_back(ability_condition_row);
    }
}
void flatten_conditional_target_query(const ProtoBufConditionalTargetQuery& conditional_target_query,
                                      int64_t batch_index, FlatConditionalTargetQuery& flat_conditional_target_query) {
    if (conditional_target_query.has_logical_query_operator()) {  // Not a leaf node
        flat_conditional_target_query.logical_operator.push_back(conditional_target_query.logical_query_operator());
    }
}

void pad_flat_conditional_target_query(FlatConditionalTargetQuery& flat) {}

void flatten_game_interaction_data(
    const google::protobuf::RepeatedPtrField<ProtoBufGameInteractionData>& game_interaction_data_list,
    int64_t batch_index, FlatGameInteractionBatch& flat) {
    flat.game_interaction_data_type_offsets.push_back(flat.game_interaction_data_types.size());
    for (const auto& game_interaction_data : game_interaction_data_list) {
        flat.game_interaction_data_types.push_back(game_interaction_data.data_type());
        switch (game_interaction_data.data_type()) {
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_MULLIGAN_DATA:
                break;  // this can be ignored, it will always be the only game interaction.
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_NUMBER_DATA:
                flat.number_data.push_back(game_interaction_data.number_data().number());
                flat.number_data_batch_index.push_back(batch_index);
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_TARGET_DATA:
                flat.target_data_batch_index.push_back(batch_index);
                flat.target_data_possible_targets_deck_ids.insert(
                    flat.target_data_possible_targets_deck_ids.end(),
                    game_interaction_data.target_data().possible_targets().begin(),
                    game_interaction_data.target_data().possible_targets().end());
                flat.target_data_possible_targets_deck_ids_length.push_back(
                    game_interaction_data.target_data().possible_targets_size());
                flat.target_data_target_action.push_back(game_interaction_data.target_data().target_action());
                flat.target_data_remainder_action.push_back(game_interaction_data.target_data().remainder_action());
                if (game_interaction_data.target_data().has_number_of_targets()) {
                    flat.target_data_number_of_targets.push_back(
                        game_interaction_data.target_data().number_of_targets());
                } else {
                    flat.target_data_number_of_targets.push_back(-1);  // padding value
                    flatten_conditional_target_query(game_interaction_data.target_data().conditional_target_query(),
                                                     batch_index, flat.flat_conditional_target_query);
                }
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA:
                flat.interaction_card_batch_index.push_back(batch_index);
                flat.interaction_card_deck_id.push_back(game_interaction_data.interaction_card_data().card());
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_ATTACK_DATA:
                append_attack_instructions_and_conditions(game_interaction_data.attack_data().attack(),
                                                          flat.instructions_and_conditions, batch_index);
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_ABILITY_DATA:
                append_ability_instructions_and_conditions(game_interaction_data.ability_data().ability(),
                                                           flat.instructions_and_conditions, batch_index);
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_WINNER_DATA:
                break;  // this can be ignored, it will always be the only game interaction.
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA:
                flat.select_from_batch_index.push_back(batch_index);
                flat.select_from.push_back(game_interaction_data.select_from_data().select_from());
                break;
            default:
                throw std::invalid_argument("Invalid game interaction data type");
        }
    }
}

FlatGameInteractionBatch flatten_game_interaction_batch(const std::vector<ProtoBufGameInteraction>& game_interactions) {
    FlatGameInteractionBatch flat;
    flat.game_interaction_types.reserve(game_interactions.size());
    flat.game_interaction_data_type_offsets.reserve(game_interactions.size());
    for (int64_t batch_index = 0; batch_index < game_interactions.size(); ++batch_index) {
        const auto& game_interaction = game_interactions[batch_index];

        flat.game_interaction_types.push_back(game_interaction.type());
        flatten_game_interaction_data(game_interaction.data(), batch_index, flat);
    }
    return flat;
}
}  // namespace

GameInteractionEmbeddingImpl::GameInteractionEmbeddingImpl(int64_t dimension_out, torch::Device device,
                                                           torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    game_interaction_type_embedding_ = register_module(
        "game_interaction_type_embedding", torch::nn::Embedding(NUMBER_GAME_INTERACTION_TYPES, dimension_out));
    to(device, dtype);
}

torch::Tensor GameInteractionEmbeddingImpl::forward(const std::vector<ProtoBufGameInteraction>& game_interactions) {
    return torch::tensor(1);
}