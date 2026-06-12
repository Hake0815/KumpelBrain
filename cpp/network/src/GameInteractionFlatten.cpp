#include "network/include/GameInteractionFlatten.h"

#include <stdexcept>

using ProtoBufGameInteractionData = gamecore::serialization::ProtoBufGameInteractionData;
using ProtoBufGameInteractionDataType = gamecore::serialization::ProtoBufGameInteractionDataType;
using ProtoBufConditionalTargetQuery = gamecore::serialization::ProtoBufConditionalTargetQuery;
using ProtoBufAbility = gamecore::serialization::ProtoBufAbility;
using ProtoBufAttack = gamecore::serialization::ProtoBufAttack;
using ProtoBufInstruction = gamecore::serialization::ProtoBufInstruction;
using ProtoBufCondition = gamecore::serialization::ProtoBufCondition;

namespace {

template <typename T>
void append_repeated_proto_to_vector(const google::protobuf::RepeatedPtrField<T>& src, std::vector<T>& dst) {
    dst.insert(dst.end(), src.begin(), src.end());
}

void append_attack_instructions_and_conditions(const ProtoBufAttack& attack,
                                               InstructionsAndConditions& instructions_and_conditions,
                                               int64_t batch_index) {
    const bool has_instructions = attack.instructions_size() > 0;
    const bool has_energy = attack.energy_cost_size() > 0;
    if (!has_instructions && !has_energy) {
        throw std::invalid_argument(
            "Invalid ProtoBufAttack in GAME_INTERACTION_DATA_TYPE_ATTACK_DATA: expected instructions or energy_cost");
    }

    const int64_t attack_slot = static_cast<int64_t>(instructions_and_conditions.instruction_attack_indices.size());
    if (has_energy) {
        for (const auto energy_type : attack.energy_cost()) {
            instructions_and_conditions.energy_flat.push_back(static_cast<int64_t>(energy_type));
            instructions_and_conditions.energy_slot_per_token.push_back(attack_slot);
        }
    }
    instructions_and_conditions.instructions.emplace_back();
    if (has_instructions) {
        append_repeated_proto_to_vector<ProtoBufInstruction>(attack.instructions(),
                                                             instructions_and_conditions.instructions.back());
    }
    instructions_and_conditions.instruction_card_parent_indices.push_back({static_cast<int>(batch_index), 0});
    instructions_and_conditions.instruction_attack_indices.push_back(
        static_cast<int64_t>(instructions_and_conditions.instructions.size() - 1));
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
        return;
    }
    if (ability.conditions_size() > 0) {
        throw std::invalid_argument(
            "Invalid ProtoBufAbility in GAME_INTERACTION_DATA_TYPE_ABILITY_DATA: conditions without instructions");
    }
    throw std::invalid_argument(
        "Invalid ProtoBufAbility in GAME_INTERACTION_DATA_TYPE_ABILITY_DATA: expected instructions");
}
int64_t append_conditional_target_query_node(const ProtoBufConditionalTargetQuery& query, int64_t depth,
                                             FlatConditionalTargetQuery& flat) {
    std::vector<int64_t> child_nodes;
    child_nodes.reserve(query.nested_queries_size());
    for (const auto& nested_query : query.nested_queries()) {
        child_nodes.push_back(append_conditional_target_query_node(nested_query, depth + 1, flat));
    }

    const auto node_id = static_cast<int64_t>(flat.node_is_leaf.size());
    const auto child_start = static_cast<int64_t>(flat.child_idx.size());
    flat.child_idx.insert(flat.child_idx.end(), child_nodes.begin(), child_nodes.end());
    flat.child_ptr.push_back(child_start);
    flat.node_depth.push_back(depth);

    const bool is_leaf = query.nested_queries_size() == 0;
    flat.node_is_leaf.push_back(is_leaf ? 1 : 0);
    if (is_leaf) {
        if (!query.has_int_range() || !query.has_selection_qualifier()) {
            throw std::invalid_argument(
                "Invalid ProtoBufConditionalTargetQuery leaf: expected int_range and selection_qualifier");
        }

        flat.node_logical_operator.push_back(-1);  // padding
        flat.leaf_node_index.push_back(node_id);
        flat.leaf_int_range.push_back(static_cast<int64_t>(query.int_range().min()));
        flat.leaf_int_range.push_back(static_cast<int64_t>(query.int_range().max()));
        flat.leaf_selection_qualifier.push_back(static_cast<int64_t>(query.selection_qualifier()));
    } else {
        if (!query.has_logical_query_operator()) {
            throw std::invalid_argument(
                "Invalid ProtoBufConditionalTargetQuery internal node: expected logical_query_operator");
        }

        flat.node_logical_operator.push_back(static_cast<int64_t>(query.logical_query_operator()));
    }

    return node_id;
}

void flatten_conditional_target_query(const ProtoBufConditionalTargetQuery& conditional_target_query,
                                      int64_t target_data_index,
                                      FlatConditionalTargetQuery& flat_conditional_target_query) {
    flat_conditional_target_query.root_node_index.push_back(
        append_conditional_target_query_node(conditional_target_query, 0, flat_conditional_target_query));
    flat_conditional_target_query.root_target_data_index.push_back(target_data_index);
}

void finalize_flat_conditional_target_query(FlatConditionalTargetQuery& flat) {
    flat.child_ptr.push_back(static_cast<int64_t>(flat.child_idx.size()));
}

void flatten_game_interaction_data(
    const google::protobuf::RepeatedPtrField<ProtoBufGameInteractionData>& game_interaction_data_list,
    int64_t batch_index, FlatGameInteractionBatch& flat) {
    flat.game_interaction_data_type_offsets.push_back(flat.game_interaction_data_types.size());
    for (const auto& game_interaction_data : game_interaction_data_list) {
        flat.game_interaction_data_types.push_back(game_interaction_data.data_type());
        switch (game_interaction_data.data_type()) {
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_MULLIGAN_DATA:
                // Payload omitted; interaction is fully described by type embedding only.
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_NUMBER_DATA:
                flat.number_data.push_back(game_interaction_data.number_data().number());
                flat.number_data_batch_index.push_back(batch_index);
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_TARGET_DATA: {
                const auto target_data_index = static_cast<int64_t>(flat.target_data_batch_index.size());
                flat.target_data_batch_index.push_back(batch_index);
                flat.target_data_possible_targets_deck_ids.insert(
                    flat.target_data_possible_targets_deck_ids.end(),
                    game_interaction_data.target_data().possible_targets().begin(),
                    game_interaction_data.target_data().possible_targets().end());
                flat.target_data_possible_targets_deck_ids_length.push_back(
                    game_interaction_data.target_data().possible_targets_size());
                flat.target_data_target_action.push_back(game_interaction_data.target_data().target_action());
                flat.target_data_remainder_action.push_back(game_interaction_data.target_data().remainder_action());
                flat.target_data_allow_multiple_times.push_back(
                    game_interaction_data.target_data().allow_multiple_times() ? 1 : 0);
                if (game_interaction_data.target_data().has_number_of_targets()) {
                    flat.target_data_number_of_targets.push_back(
                        game_interaction_data.target_data().number_of_targets());
                } else {
                    if (!game_interaction_data.target_data().has_conditional_target_query()) {
                        throw std::invalid_argument(
                            "Invalid ProtoBufTargetData: expected conditional_target_query or number_of_targets");
                    }
                    flat.target_data_number_of_targets.push_back(-1);  // padding value
                    flatten_conditional_target_query(game_interaction_data.target_data().conditional_target_query(),
                                                     target_data_index, flat.flat_conditional_target_query);
                }
                break;
            }
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
                // Payload omitted; interaction is fully described by type embedding only.
                break;
            case ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA:
                flat.select_from_batch_index.push_back(batch_index);
                flat.select_from.push_back(game_interaction_data.select_from_data().select_from());
                break;
            default:
                throw std::invalid_argument("Invalid game interaction data type");
        }
    }
}

}  // namespace

FlatGameInteractionBatch flatten_game_interaction_batch(
    const std::vector<std::vector<ProtoBufGameInteraction>>& game_interactions_per_game) {
    FlatGameInteractionBatch flat;
    const int64_t num_games = static_cast<int64_t>(game_interactions_per_game.size());
    flat.interaction_segment_offsets.reserve(static_cast<size_t>(num_games + 1));
    flat.interaction_segment_offsets.push_back(0);
    int64_t total_interactions = 0;
    for (int64_t game_index = 0; game_index < num_games; ++game_index) {
        const auto& game_interactions = game_interactions_per_game[static_cast<size_t>(game_index)];
        flat.game_interaction_types.reserve(flat.game_interaction_types.size() +
                                            static_cast<size_t>(game_interactions.size()));
        flat.game_interaction_data_type_offsets.reserve(flat.game_interaction_data_type_offsets.size() +
                                                        static_cast<size_t>(game_interactions.size()));
        for (int64_t local_index = 0; local_index < static_cast<int64_t>(game_interactions.size()); ++local_index) {
            const int64_t batch_index = total_interactions + local_index;
            const auto& game_interaction = game_interactions[static_cast<size_t>(local_index)];
            flat.interaction_game_indices.push_back(game_index);
            flat.game_interaction_types.push_back(game_interaction.type());
            flatten_game_interaction_data(game_interaction.data(), batch_index, flat);
        }
        total_interactions += static_cast<int64_t>(game_interactions.size());
        flat.interaction_segment_offsets.push_back(total_interactions);
    }
    finalize_flat_conditional_target_query(flat.flat_conditional_target_query);
    return flat;
}

FlatGameInteractionBatchTensors flat_game_interaction_batch_to_tensors(const FlatGameInteractionBatch& flat,
                                                                       torch::Device device) {
    const auto& q = flat.flat_conditional_target_query;

    const size_t n_interaction_game_indices = flat.interaction_game_indices.size();
    const size_t n_interaction_segment_offsets = flat.interaction_segment_offsets.size();
    const size_t n_game_interaction_types = flat.game_interaction_types.size();
    const size_t n_game_interaction_data_types = flat.game_interaction_data_types.size();
    const size_t n_game_interaction_data_type_offsets = flat.game_interaction_data_type_offsets.size();
    const size_t n_number_data_batch_index = flat.number_data_batch_index.size();
    const size_t n_number_data = flat.number_data.size();
    const size_t n_target_data_batch_index = flat.target_data_batch_index.size();
    const size_t n_target_data_possible_targets_deck_ids = flat.target_data_possible_targets_deck_ids.size();
    const size_t n_target_data_possible_targets_deck_ids_length =
        flat.target_data_possible_targets_deck_ids_length.size();
    const size_t n_target_data_target_action = flat.target_data_target_action.size();
    const size_t n_target_data_remainder_action = flat.target_data_remainder_action.size();
    const size_t n_target_data_number_of_targets = flat.target_data_number_of_targets.size();
    const size_t n_target_data_allow_multiple_times = flat.target_data_allow_multiple_times.size();
    const size_t n_interaction_card_batch_index = flat.interaction_card_batch_index.size();
    const size_t n_interaction_card_deck_id = flat.interaction_card_deck_id.size();
    const size_t n_select_from_batch_index = flat.select_from_batch_index.size();
    const size_t n_select_from = flat.select_from.size();

    const size_t n_node_is_leaf = q.node_is_leaf.size();
    const size_t n_node_logical_operator = q.node_logical_operator.size();
    const size_t n_node_depth = q.node_depth.size();
    const size_t n_child_ptr = q.child_ptr.size();
    const size_t n_child_idx = q.child_idx.size();
    const size_t n_leaf_node_index = q.leaf_node_index.size();
    const size_t n_leaf_int_range = q.leaf_int_range.size();
    const size_t n_leaf_selection_qualifier = q.leaf_selection_qualifier.size();
    const size_t n_root_node_index = q.root_node_index.size();
    const size_t n_root_target_data_index = q.root_target_data_index.size();

    const size_t total_int64 =
        n_interaction_game_indices + n_interaction_segment_offsets + n_game_interaction_types +
        n_game_interaction_data_types + n_game_interaction_data_type_offsets +
        n_number_data_batch_index + n_number_data + n_target_data_batch_index +
        n_target_data_possible_targets_deck_ids + n_target_data_possible_targets_deck_ids_length +
        n_target_data_target_action + n_target_data_remainder_action + n_target_data_number_of_targets +
        n_target_data_allow_multiple_times + n_interaction_card_batch_index + n_interaction_card_deck_id +
        n_select_from_batch_index + n_select_from +
        n_node_is_leaf + n_node_logical_operator + n_node_depth + n_child_ptr + n_child_idx + n_leaf_node_index +
        n_leaf_int_range + n_leaf_selection_qualifier + n_root_node_index + n_root_target_data_index;

    std::vector<int64_t> int64_host;
    int64_host.reserve(total_int64);

    const auto push_block = [&](const std::vector<int64_t>& v) {
        int64_host.insert(int64_host.end(), v.begin(), v.end());
    };

    int64_t off = 0;
    const int64_t off_interaction_game_indices = off;
    push_block(flat.interaction_game_indices);
    off += static_cast<int64_t>(n_interaction_game_indices);
    const int64_t off_interaction_segment_offsets = off;
    push_block(flat.interaction_segment_offsets);
    off += static_cast<int64_t>(n_interaction_segment_offsets);
    const int64_t off_game_interaction_types = off;
    push_block(flat.game_interaction_types);
    off += static_cast<int64_t>(n_game_interaction_types);
    const int64_t off_game_interaction_data_types = off;
    push_block(flat.game_interaction_data_types);
    off += static_cast<int64_t>(n_game_interaction_data_types);
    const int64_t off_game_interaction_data_type_offsets = off;
    push_block(flat.game_interaction_data_type_offsets);
    off += static_cast<int64_t>(n_game_interaction_data_type_offsets);
    const int64_t off_number_data_batch_index = off;
    push_block(flat.number_data_batch_index);
    off += static_cast<int64_t>(n_number_data_batch_index);
    const int64_t off_number_data = off;
    push_block(flat.number_data);
    off += static_cast<int64_t>(n_number_data);
    const int64_t off_target_data_batch_index = off;
    push_block(flat.target_data_batch_index);
    off += static_cast<int64_t>(n_target_data_batch_index);
    const int64_t off_target_data_possible_targets_deck_ids = off;
    push_block(flat.target_data_possible_targets_deck_ids);
    off += static_cast<int64_t>(n_target_data_possible_targets_deck_ids);
    const int64_t off_target_data_possible_targets_deck_ids_length = off;
    push_block(flat.target_data_possible_targets_deck_ids_length);
    off += static_cast<int64_t>(n_target_data_possible_targets_deck_ids_length);
    const int64_t off_target_data_target_action = off;
    push_block(flat.target_data_target_action);
    off += static_cast<int64_t>(n_target_data_target_action);
    const int64_t off_target_data_remainder_action = off;
    push_block(flat.target_data_remainder_action);
    off += static_cast<int64_t>(n_target_data_remainder_action);
    const int64_t off_target_data_number_of_targets = off;
    push_block(flat.target_data_number_of_targets);
    off += static_cast<int64_t>(n_target_data_number_of_targets);
    const int64_t off_target_data_allow_multiple_times = off;
    push_block(flat.target_data_allow_multiple_times);
    off += static_cast<int64_t>(n_target_data_allow_multiple_times);
    const int64_t off_interaction_card_batch_index = off;
    push_block(flat.interaction_card_batch_index);
    off += static_cast<int64_t>(n_interaction_card_batch_index);
    const int64_t off_interaction_card_deck_id = off;
    push_block(flat.interaction_card_deck_id);
    off += static_cast<int64_t>(n_interaction_card_deck_id);
    const int64_t off_select_from_batch_index = off;
    push_block(flat.select_from_batch_index);
    off += static_cast<int64_t>(n_select_from_batch_index);
    const int64_t off_select_from = off;
    push_block(flat.select_from);
    off += static_cast<int64_t>(n_select_from);

    const int64_t off_node_is_leaf = off;
    push_block(q.node_is_leaf);
    off += static_cast<int64_t>(n_node_is_leaf);
    const int64_t off_node_logical_operator = off;
    push_block(q.node_logical_operator);
    off += static_cast<int64_t>(n_node_logical_operator);
    const int64_t off_node_depth = off;
    push_block(q.node_depth);
    off += static_cast<int64_t>(n_node_depth);
    const int64_t off_child_ptr = off;
    push_block(q.child_ptr);
    off += static_cast<int64_t>(n_child_ptr);
    const int64_t off_child_idx = off;
    push_block(q.child_idx);
    off += static_cast<int64_t>(n_child_idx);
    const int64_t off_leaf_node_index = off;
    push_block(q.leaf_node_index);
    off += static_cast<int64_t>(n_leaf_node_index);
    const int64_t off_leaf_int_range = off;
    push_block(q.leaf_int_range);
    off += static_cast<int64_t>(n_leaf_int_range);
    const int64_t off_leaf_selection_qualifier = off;
    push_block(q.leaf_selection_qualifier);
    off += static_cast<int64_t>(n_leaf_selection_qualifier);
    const int64_t off_root_node_index = off;
    push_block(q.root_node_index);
    off += static_cast<int64_t>(n_root_node_index);
    const int64_t off_root_target_data_index = off;
    push_block(q.root_target_data_index);
    off += static_cast<int64_t>(n_root_target_data_index);

    const auto index_options = torch::TensorOptions().device(device).dtype(torch::kLong);
    auto int64_buf = torch::tensor(int64_host, index_options);

    FlatGameInteractionBatchTensors out;
    out.interaction_game_indices =
        int64_buf.narrow(0, off_interaction_game_indices, static_cast<int64_t>(n_interaction_game_indices));
    out.interaction_segment_offsets =
        int64_buf.narrow(0, off_interaction_segment_offsets, static_cast<int64_t>(n_interaction_segment_offsets));
    out.game_interaction_types =
        int64_buf.narrow(0, off_game_interaction_types, static_cast<int64_t>(n_game_interaction_types));
    out.game_interaction_data_types =
        int64_buf.narrow(0, off_game_interaction_data_types, static_cast<int64_t>(n_game_interaction_data_types));
    out.game_interaction_data_type_offsets = int64_buf.narrow(
        0, off_game_interaction_data_type_offsets, static_cast<int64_t>(n_game_interaction_data_type_offsets));
    out.number_data_batch_index =
        int64_buf.narrow(0, off_number_data_batch_index, static_cast<int64_t>(n_number_data_batch_index));
    out.number_data = int64_buf.narrow(0, off_number_data, static_cast<int64_t>(n_number_data));
    out.target_data_batch_index =
        int64_buf.narrow(0, off_target_data_batch_index, static_cast<int64_t>(n_target_data_batch_index));
    out.target_data_possible_targets_deck_ids = int64_buf.narrow(
        0, off_target_data_possible_targets_deck_ids, static_cast<int64_t>(n_target_data_possible_targets_deck_ids));
    out.target_data_possible_targets_deck_ids_length =
        int64_buf.narrow(0, off_target_data_possible_targets_deck_ids_length,
                         static_cast<int64_t>(n_target_data_possible_targets_deck_ids_length));
    out.target_data_target_action =
        int64_buf.narrow(0, off_target_data_target_action, static_cast<int64_t>(n_target_data_target_action));
    out.target_data_remainder_action =
        int64_buf.narrow(0, off_target_data_remainder_action, static_cast<int64_t>(n_target_data_remainder_action));
    out.target_data_number_of_targets =
        int64_buf.narrow(0, off_target_data_number_of_targets, static_cast<int64_t>(n_target_data_number_of_targets));
    out.target_data_allow_multiple_times = int64_buf.narrow(
        0, off_target_data_allow_multiple_times, static_cast<int64_t>(n_target_data_allow_multiple_times));
    out.interaction_card_batch_index =
        int64_buf.narrow(0, off_interaction_card_batch_index, static_cast<int64_t>(n_interaction_card_batch_index));
    out.interaction_card_deck_id =
        int64_buf.narrow(0, off_interaction_card_deck_id, static_cast<int64_t>(n_interaction_card_deck_id));
    out.select_from_batch_index =
        int64_buf.narrow(0, off_select_from_batch_index, static_cast<int64_t>(n_select_from_batch_index));
    out.select_from = int64_buf.narrow(0, off_select_from, static_cast<int64_t>(n_select_from));

    auto& qt = out.flat_conditional_target_query;
    qt.node_is_leaf = int64_buf.narrow(0, off_node_is_leaf, static_cast<int64_t>(n_node_is_leaf));
    qt.node_logical_operator =
        int64_buf.narrow(0, off_node_logical_operator, static_cast<int64_t>(n_node_logical_operator));
    qt.node_depth = int64_buf.narrow(0, off_node_depth, static_cast<int64_t>(n_node_depth));
    qt.child_ptr = int64_buf.narrow(0, off_child_ptr, static_cast<int64_t>(n_child_ptr));
    qt.child_idx = int64_buf.narrow(0, off_child_idx, static_cast<int64_t>(n_child_idx));
    qt.leaf_node_index = int64_buf.narrow(0, off_leaf_node_index, static_cast<int64_t>(n_leaf_node_index));
    const auto num_leaves = static_cast<int64_t>(n_leaf_node_index);
    qt.leaf_int_range =
        int64_buf.narrow(0, off_leaf_int_range, static_cast<int64_t>(n_leaf_int_range)).view({num_leaves, 2});
    qt.leaf_selection_qualifier =
        int64_buf.narrow(0, off_leaf_selection_qualifier, static_cast<int64_t>(n_leaf_selection_qualifier));
    qt.root_node_index = int64_buf.narrow(0, off_root_node_index, static_cast<int64_t>(n_root_node_index));
    qt.root_target_data_index =
        int64_buf.narrow(0, off_root_target_data_index, static_cast<int64_t>(n_root_target_data_index));

    out.instructions_and_conditions = flat.instructions_and_conditions;
    return out;
}
