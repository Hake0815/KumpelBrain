#include "network/include/GameInteractionEmbedding.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "network/include/AttentionUtils.h"
#include "network/include/SharedConstants.h"

using ProtoBufGameInteractionData = gamecore::serialization::ProtoBufGameInteractionData;
using ProtoBufGameInteractionDataType = gamecore::serialization::ProtoBufGameInteractionDataType;
using ProtoBufConditionalTargetQuery = gamecore::serialization::ProtoBufConditionalTargetQuery;
using ProtoBufAbility = gamecore::serialization::ProtoBufAbility;
namespace {

torch::Tensor tensor_from_int64_vector(const std::vector<int64_t>& values, torch::Device device) {
    return torch::tensor(values, torch::TensorOptions().device(device).dtype(torch::kLong));
}

torch::Tensor tensor_from_bool_vector(const std::vector<int64_t>& values, torch::Device device) {
    return torch::tensor(values, torch::TensorOptions().device(device).dtype(torch::kBool));
}

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

        flat.node_logical_operator.push_back(-1); // padding
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
                break;  // this can be ignored, it will always be the only game interaction.
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
    finalize_flat_conditional_target_query(flat.flat_conditional_target_query);
    return flat;
}
}  // namespace

GameInteractionEmbeddingImpl::GameInteractionEmbeddingImpl(int64_t dimension_out, torch::Device device,
                                                           torch::Dtype dtype)
    : dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    game_interaction_type_embedding_ = register_module(
        "game_interaction_type_embedding", torch::nn::Embedding(NUMBER_GAME_INTERACTION_TYPES, dimension_out));
    conditional_query_int_range_embedding_ =
        register_module("conditional_query_int_range_embedding",
                        NormalizedLinear(2, dimension_out, static_cast<double>(DECK_SIZE), device, dtype));
    conditional_query_selection_qualifier_embedding_ =
        register_module("conditional_query_selection_qualifier_embedding",
                        torch::nn::Embedding(NUMBER_SELECTION_QUALIFIERS, dimension_out));
    conditional_query_leaf_projection_ =
        register_module("conditional_query_leaf_projection", torch::nn::Linear(2 * dimension_out, dimension_out));
    conditional_query_operator_embedding_ = register_module(
        "conditional_query_operator_embedding", torch::nn::Embedding(NUMBER_LOGICAL_QUERY_OPERATORS, dimension_out));
    conditional_query_attention_ =
        register_module("conditional_query_attention",
                        MultiHeadAttention(dimension_out, dimension_out, dimension_out,
                                           std::max<int64_t>(dimension_out / 16, 1), 2, 0.0, false, device, dtype));
    to(device, dtype);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_conditional_target_queries(const FlatConditionalTargetQuery& flat) {
    if (flat.root_node_index.empty()) {
        return torch::zeros({0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
    }

    const auto num_nodes = static_cast<int64_t>(flat.node_is_leaf.size());
    auto node_embeddings =
        torch::zeros({num_nodes, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));

    const auto node_is_leaf = tensor_from_bool_vector(flat.node_is_leaf, device_);
    const auto node_logical_operator = tensor_from_int64_vector(flat.node_logical_operator, device_);
    const auto node_depth = tensor_from_int64_vector(flat.node_depth, device_);
    const auto child_ptr = tensor_from_int64_vector(flat.child_ptr, device_);
    const auto child_idx = tensor_from_int64_vector(flat.child_idx, device_);

    if (!flat.leaf_node_index.empty()) {
        const auto leaf_node_index = tensor_from_int64_vector(flat.leaf_node_index, device_);
        auto leaf_int_range = tensor_from_int64_vector(flat.leaf_int_range, device_)
                                  .view({static_cast<int64_t>(flat.leaf_node_index.size()), 2})
                                  .to(dtype_);
        const auto leaf_selection_qualifier = tensor_from_int64_vector(flat.leaf_selection_qualifier, device_);

        auto range_embeddings = conditional_query_int_range_embedding_->forward(leaf_int_range);
        auto qualifier_embeddings = conditional_query_selection_qualifier_embedding_(leaf_selection_qualifier);
        auto leaf_embeddings =
            conditional_query_leaf_projection_->forward(torch::cat({range_embeddings, qualifier_embeddings}, 1));
        node_embeddings.index_copy_(0, leaf_node_index, leaf_embeddings);
    }

    const auto internal_mask = torch::logical_not(node_is_leaf);
    if (internal_mask.any().item<bool>()) {
        const auto max_depth = node_depth.max().item<int64_t>();
        for (int64_t depth = max_depth; depth >= 0; --depth) {
            const auto depth_mask = torch::logical_and(internal_mask, node_depth.eq(depth));
            const auto reduce_nodes = torch::nonzero(depth_mask).squeeze(1);
            if (reduce_nodes.numel() == 0) {
                continue;
            }

            const auto child_start = child_ptr.index_select(0, reduce_nodes);
            const auto child_end = child_ptr.index_select(0, reduce_nodes + 1);
            const auto child_count = child_end - child_start;
            const auto max_children = child_count.max().item<int64_t>();
            const auto positions =
                torch::arange(max_children, torch::TensorOptions().device(device_).dtype(torch::kLong));
            const auto valid_children = positions.unsqueeze(0) < child_count.unsqueeze(1);
            const auto gather_positions = child_start.unsqueeze(1) + positions.unsqueeze(0);
            const auto safe_positions = gather_positions.masked_fill(torch::logical_not(valid_children), 0).reshape(-1);
            const auto child_indices =
                child_idx.index_select(0, safe_positions).view({reduce_nodes.size(0), max_children});
            auto child_embeddings = node_embeddings.index_select(0, child_indices.reshape(-1))
                                        .view({reduce_nodes.size(0), max_children, dimension_out_});
            child_embeddings = child_embeddings * valid_children.unsqueeze(-1).to(child_embeddings.dtype());

            auto operator_embeddings =
                conditional_query_operator_embedding_(node_logical_operator.index_select(0, reduce_nodes)).unsqueeze(1);
            auto reduced = attention_utils::query_sum_attention_pooling(
                conditional_query_attention_, operator_embeddings, child_embeddings, valid_children);
            node_embeddings.index_copy_(0, reduce_nodes, reduced);
        }
    }

    const auto root_node_index = tensor_from_int64_vector(flat.root_node_index, device_);
    return node_embeddings.index_select(0, root_node_index);
}

torch::Tensor GameInteractionEmbeddingImpl::forward(const std::vector<ProtoBufGameInteraction>& game_interactions) {
    auto flat = flatten_game_interaction_batch(game_interactions);
    auto target_data_embeddings =
        torch::zeros({static_cast<int64_t>(flat.target_data_batch_index.size()), dimension_out_},
                     torch::TensorOptions().device(device_).dtype(dtype_));
    auto conditional_query_embeddings = embed_conditional_target_queries(flat.flat_conditional_target_query);
    if (conditional_query_embeddings.numel() > 0) {
        auto target_data_index =
            tensor_from_int64_vector(flat.flat_conditional_target_query.root_target_data_index, device_);
        target_data_embeddings.index_copy_(0, target_data_index, conditional_query_embeddings);
    }
    return target_data_embeddings;
}