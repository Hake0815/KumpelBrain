#include "network/include/GameInteractionEmbedding.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "network/include/AttentionUtils.h"
#include "network/include/DeckIdResolution.h"
#include "network/include/EnergyTypeEmbedding.h"
#include "network/include/GameInteractionFlatten.h"
#include "network/include/SharedConstants.h"
#include "network/include/TensorUtils.h"

using ProtoBufGameInteractionDataType = gamecore::serialization::ProtoBufGameInteractionDataType;

GameInteractionEmbeddingImpl::GameInteractionEmbeddingImpl(
    std::shared_ptr<SharedEmbeddingHolderImpl> shared_embedding_holder, int64_t dimension_out,
    const SharedInstructionEmbeddings& shared_instruction_embeddings, torch::Device device, torch::Dtype dtype)
    : shared_embedding_holder_(shared_embedding_holder), dimension_out_(dimension_out), device_(device), dtype_(dtype) {
    instruction_data_embedding_ = shared_instruction_embeddings.instruction_data_embedding;
    instruction_embedding_ = shared_instruction_embeddings.instruction_embedding;
    condition_embedding_ = shared_instruction_embeddings.condition_embedding;
    attack_embedding_ = shared_instruction_embeddings.attack_embedding;
    ability_embedding_ = shared_instruction_embeddings.ability_embedding;
    register_game_interaction_specific_modules(device, dtype);
    to(device, dtype);
}

void GameInteractionEmbeddingImpl::register_game_interaction_specific_modules(torch::Device device,
                                                                              torch::Dtype dtype) {
    mask_tensor_options_ = torch::TensorOptions().device(device_).dtype(torch::kBool);
    index_tensor_options_ = torch::TensorOptions().device(device_).dtype(torch::kInt64);
    float_tensor_options_ = torch::TensorOptions().device(device_).dtype(dtype_);
    game_interaction_type_embedding_ = register_module(
        "game_interaction_type_embedding", torch::nn::Embedding(NUMBER_GAME_INTERACTION_TYPES, dimension_out_));
    game_interaction_data_type_embedding_ =
        register_module("game_interaction_data_type_embedding",
                        torch::nn::Embedding(NUMBER_GAME_INTERACTION_DATA_TYPES, dimension_out_));
    conditional_query_int_range_embedding_ =
        register_module("conditional_query_int_range_embedding",
                        NormalizedLinear(2, dimension_out_, static_cast<double>(DECK_SIZE), device, dtype));
    conditional_query_selection_qualifier_embedding_ =
        register_module("conditional_query_selection_qualifier_embedding",
                        torch::nn::Embedding(NUMBER_SELECTION_QUALIFIERS, dimension_out_));
    conditional_query_leaf_projection_ =
        register_module("conditional_query_leaf_projection", torch::nn::Linear(2 * dimension_out_, dimension_out_));
    conditional_query_operator_embedding_ = register_module(
        "conditional_query_operator_embedding", torch::nn::Embedding(NUMBER_LOGICAL_QUERY_OPERATORS, dimension_out_));
    conditional_query_attention_ =
        register_module("conditional_query_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 1), 2, 0.0, false, device, dtype));
    target_data_attention_ =
        register_module("target_data_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 1), 4, 0.0, false, device, dtype));
    game_interaction_attention_ =
        register_module("game_interaction_attention",
                        MultiHeadAttention(dimension_out_, dimension_out_, dimension_out_,
                                           std::max<int64_t>(dimension_out_ / 16, 1), 6, 0.0, false, device, dtype));
    action_on_selection_embedding_ = register_module("action_on_selection_embedding",
                                                     torch::nn::Embedding(NUMBER_ACTION_ON_SELECTION, dimension_out_));
    target_data_addition_embedding_ =
        register_module("target_data_addition_embedding", torch::nn::Embedding(1, dimension_out_));
    remainder_data_addition_embedding_ =
        register_module("remainder_data_addition_embedding", torch::nn::Embedding(1, dimension_out_));
    number_of_targets_embedding_ =
        register_module("number_of_targets_embedding", NormalizedLinear(1, dimension_out_, 5.0, device, dtype));
    number_data_embedding_ =
        register_module("number_data_embedding", NormalizedLinear(1, dimension_out_, 5.0, device, dtype));
    select_from_embedding_ =
        register_module("select_from_embedding", torch::nn::Embedding(NUMBER_SELECT_FROM, dimension_out_));
    scalar_zero_index_ = torch::zeros({1}, index_tensor_options_);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_target_action(const torch::Tensor& target_action) {
    return action_on_selection_embedding_(target_action) + target_data_addition_embedding_(scalar_zero_index_);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_remainder_action(const torch::Tensor& remainder_action) {
    return action_on_selection_embedding_(remainder_action) +
           remainder_data_addition_embedding_(scalar_zero_index_);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_conditional_target_queries(
    const FlatConditionalTargetQueryTensors& flat) {
    if (flat.root_node_index.numel() == 0) {
        return torch::zeros({0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
    }

    const auto num_nodes = flat.node_is_leaf.size(0);
    auto node_embeddings =
        torch::zeros({num_nodes, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));

    const auto node_is_leaf = flat.node_is_leaf.ne(0);
    const auto& node_logical_operator = flat.node_logical_operator;
    const auto& node_depth = flat.node_depth;
    const auto& child_ptr = flat.child_ptr;
    const auto& child_idx = flat.child_idx;

    if (flat.leaf_node_index.numel() > 0) {
        const auto& leaf_node_index = flat.leaf_node_index;
        const auto leaf_int_range = flat.leaf_int_range.to(dtype_);
        const auto& leaf_selection_qualifier = flat.leaf_selection_qualifier;

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

    return node_embeddings.index_select(0, flat.root_node_index);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_target_data(
    const FlatGameInteractionBatchTensors& flat_game_interaction_batch, torch::Tensor card_indices,
    torch::Tensor cards) {
    const auto num_target_data = flat_game_interaction_batch.target_data_target_action.size(0);
    if (num_target_data == 0) {
        return torch::empty({0, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));
    }

    auto embedded_selection_target_data =
        torch::zeros({num_target_data, dimension_out_}, torch::TensorOptions().device(device_).dtype(dtype_));

    const auto number_of_targets = flat_game_interaction_batch.target_data_number_of_targets;
    const auto has_number_of_targets_mask = number_of_targets.ne(-1);
    if (has_number_of_targets_mask.any().item<bool>()) {
        const auto number_of_targets_indices = torch::nonzero(has_number_of_targets_mask).squeeze(1);
        const auto number_of_targets_values =
            number_of_targets.index_select(0, number_of_targets_indices).to(dtype_).unsqueeze(1);
        const auto embedded_number_of_targets = number_of_targets_embedding_->forward(number_of_targets_values);
        embedded_selection_target_data.index_copy_(0, number_of_targets_indices, embedded_number_of_targets);
    }

    const auto embedded_conditional_query =
        embed_conditional_target_queries(flat_game_interaction_batch.flat_conditional_target_query);
    if (embedded_conditional_query.numel() > 0) {
        const auto& conditional_target_data_index =
            flat_game_interaction_batch.flat_conditional_target_query.root_target_data_index;
        embedded_selection_target_data.index_copy_(0, conditional_target_data_index, embedded_conditional_query);
    }

    const auto embedded_target_actions = embed_target_action(flat_game_interaction_batch.target_data_target_action);
    const auto embedded_remainder_actions =
        embed_remainder_action(flat_game_interaction_batch.target_data_remainder_action);

    const auto possible_targets_cards = resolve_cards_from_deck_ids(
        flat_game_interaction_batch.target_data_possible_targets_deck_ids, card_indices, cards,
        "embedGameInteraction target_data possible_targets");

    const auto& lengths = flat_game_interaction_batch.target_data_possible_targets_deck_ids_length;
    const auto max_possible_targets = lengths.max().item<int64_t>();
    const auto possible_targets_offsets = torch::cat({torch::zeros({1}, lengths.options()), lengths.cumsum(0)}, 0);
    auto [padded_possible_targets, possible_targets_mask] = tensor_utils::pad_by_offsets(
        possible_targets_cards, possible_targets_offsets, dimension_out_, max_possible_targets);

    const auto prefix =
        torch::stack({embedded_target_actions, embedded_remainder_actions, embedded_selection_target_data}, 1);
    const auto padded_sequences = torch::cat({prefix, padded_possible_targets}, 1);
    const auto mask_options = torch::TensorOptions().device(device_).dtype(torch::kBool);
    const auto valid_token_mask =
        torch::cat({torch::ones({num_target_data, 3}, mask_options), possible_targets_mask}, 1);

    return attention_utils::masked_self_attention_reduce(target_data_attention_, padded_sequences, valid_token_mask);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_attack_energy_costs(
    const InstructionsAndConditions& instructions_and_conditions) {
    if (instructions_and_conditions.energy_flat.empty()) {
        return torch::empty({0, dimension_out_}, float_tensor_options_);
    }
    const auto num_energy = static_cast<int64_t>(instructions_and_conditions.energy_flat.size());
    auto energy_types = torch::tensor(instructions_and_conditions.energy_flat, index_tensor_options_);
    auto contexts = torch::full({num_energy}, static_cast<int64_t>(ATTACK_COST), index_tensor_options_);
    return shared_embedding_holder_->energy_type_embedding_->forward(energy_types, contexts);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_attacks(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_attack_indices, const torch::Tensor& attack_energy_costs,
    const std::vector<int64_t>& energy_slot_per_token) {
    if (instruction_attack_indices.empty()) {
        return torch::empty({0, dimension_out_}, float_tensor_options_);
    }

    const int64_t num_attacks = static_cast<int64_t>(instruction_attack_indices.size());
    auto attack_energy_sums = torch::zeros({num_attacks, dimension_out_}, float_tensor_options_);
    if (attack_energy_costs.size(0) > 0) {
        auto slot_idx = torch::tensor(energy_slot_per_token, index_tensor_options_);
        attack_energy_sums.index_add_(0, slot_idx, attack_energy_costs);
    }

    const auto attack_instruction_rows = torch::tensor(instruction_attack_indices, index_tensor_options_);
    auto embedded_instruction_attacks = embedded_instructions_pair.first.index_select(0, attack_instruction_rows);
    auto embedded_instruction_attacks_mask = embedded_instructions_pair.second.index_select(0, attack_instruction_rows);
    return attack_embedding_->forward(attack_energy_sums, embedded_instruction_attacks,
                                      embedded_instruction_attacks_mask);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_ability(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_ability_indices,
    const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
    const std::vector<int64_t>& ability_condition_row_for_instruction_ability) {
    if (instruction_ability_indices.empty()) {
        return torch::empty({0, dimension_out_}, float_tensor_options_);
    }

    auto instruction_index_tensor = torch::tensor(instruction_ability_indices, index_tensor_options_);
    auto embedded_instruction_abilities = embedded_instructions_pair.first.index_select(0, instruction_index_tensor);
    auto embedded_instruction_abilities_mask =
        embedded_instructions_pair.second.index_select(0, instruction_index_tensor);

    const int64_t number_of_abilities = static_cast<int64_t>(instruction_ability_indices.size());

    const bool any_condition =
        std::any_of(ability_condition_row_for_instruction_ability.begin(),
                    ability_condition_row_for_instruction_ability.end(), [](int64_t v) { return v >= 0; });
    const int64_t max_number_of_conditions = any_condition ? embedded_conditions_pair.first.size(1) : 0;

    auto cond_vals =
        torch::zeros({number_of_abilities, max_number_of_conditions, dimension_out_}, float_tensor_options_);
    auto cond_mask = torch::zeros({number_of_abilities, max_number_of_conditions}, mask_tensor_options_);
    if (max_number_of_conditions > 0) {
        std::vector<int64_t> valid_dst, valid_src;
        for (int64_t i = 0; i < number_of_abilities; ++i) {
            const int64_t cidx = ability_condition_row_for_instruction_ability[static_cast<size_t>(i)];
            if (cidx >= 0) {
                valid_dst.push_back(i);
                valid_src.push_back(cidx);
            }
        }
        if (!valid_dst.empty()) {
            auto src_t = torch::tensor(valid_src, index_tensor_options_);
            auto dst_t = torch::tensor(valid_dst, index_tensor_options_);
            cond_vals.index_put_({dst_t}, embedded_conditions_pair.first.index_select(0, src_t));
            cond_mask.index_put_({dst_t}, embedded_conditions_pair.second.index_select(0, src_t));
        }
    }

    return ability_embedding_->forward(embedded_instruction_abilities, embedded_instruction_abilities_mask, cond_vals,
                                       cond_mask);
}

torch::Tensor GameInteractionEmbeddingImpl::embed_number_data(const torch::Tensor& number_data) {
    if (number_data.numel() == 0) {
        return torch::empty({0, dimension_out_}, float_tensor_options_);
    }
    return number_data_embedding_->forward(number_data.to(dtype_).unsqueeze(1));
}

namespace {

void index_copy_payload_for_data_type(torch::Tensor& payload, const torch::Tensor& data_types, int64_t data_type_value,
                                      const torch::Tensor& values) {
    const auto positions = torch::nonzero(data_types.eq(data_type_value)).squeeze(1);
    if (positions.numel() == 0 || values.numel() == 0) {
        return;
    }
    payload.index_copy_(0, positions, values);
}

}  // namespace

torch::Tensor GameInteractionEmbeddingImpl::embed_interaction_data(const FlatGameInteractionBatchTensors& flat_tensors,
                                                                   const torch::Tensor& embedded_target_data,
                                                                   const torch::Tensor& embedded_number_data,
                                                                   const torch::Tensor& interaction_card_data,
                                                                   const torch::Tensor& embedded_attack_data,
                                                                   const torch::Tensor& embedded_ability_data,
                                                                   const torch::Tensor& embedded_select_from) {
    const auto& data_types = flat_tensors.game_interaction_data_types;
    const auto num_data_rows = data_types.size(0);
    if (num_data_rows == 0) {
        return torch::empty({0, dimension_out_}, float_tensor_options_);
    }

    auto payload = torch::zeros({num_data_rows, dimension_out_}, float_tensor_options_);

    index_copy_payload_for_data_type(
        payload, data_types,
        static_cast<int64_t>(ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_NUMBER_DATA),
        embedded_number_data);
    index_copy_payload_for_data_type(
        payload, data_types,
        static_cast<int64_t>(ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_TARGET_DATA),
        embedded_target_data);
    index_copy_payload_for_data_type(
        payload, data_types,
        static_cast<int64_t>(ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_INTERACTION_CARD_DATA),
        interaction_card_data);
    index_copy_payload_for_data_type(
        payload, data_types,
        static_cast<int64_t>(ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_SELECT_FROM_DATA),
        embedded_select_from);

    const auto attack_positions =
        torch::nonzero(data_types.eq(static_cast<int64_t>(
                           ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_ATTACK_DATA)))
            .squeeze(1);
    if (attack_positions.numel() > 0) {
        if (embedded_attack_data.size(0) == 0) {
            throw std::invalid_argument(
                "GAME_INTERACTION_DATA_TYPE_ATTACK_DATA present but produced no attack embedding");
        }
        payload.index_copy_(0, attack_positions, embedded_attack_data);
    }

    const auto ability_positions =
        torch::nonzero(data_types.eq(static_cast<int64_t>(
                           ProtoBufGameInteractionDataType::GAME_INTERACTION_DATA_TYPE_ABILITY_DATA)))
            .squeeze(1);
    if (ability_positions.numel() > 0) {
        if (embedded_ability_data.size(0) == 0) {
            throw std::invalid_argument(
                "GAME_INTERACTION_DATA_TYPE_ABILITY_DATA present but produced no ability embedding");
        }
        payload.index_copy_(0, ability_positions, embedded_ability_data);
    }

    return payload + game_interaction_data_type_embedding_(data_types);
}

torch::Tensor GameInteractionEmbeddingImpl::reduce_game_interactions(
    const FlatGameInteractionBatchTensors& flat_tensors, const torch::Tensor& embedded_interaction_data) {
    const auto batch_size = flat_tensors.game_interaction_types.size(0);
    if (batch_size == 0) {
        return torch::empty({0, dimension_out_}, float_tensor_options_);
    }

    const auto num_data_rows = embedded_interaction_data.size(0);
    const auto offsets_with_end =
        torch::cat({flat_tensors.game_interaction_data_type_offsets,
                    torch::tensor({num_data_rows}, flat_tensors.game_interaction_data_type_offsets.options())},
                   0);
    const auto lengths = offsets_with_end.slice(0, 1, batch_size + 1) - offsets_with_end.slice(0, 0, batch_size);
    const auto max_data_per_interaction = lengths.max().item<int64_t>();

    auto [padded_data, data_mask] = tensor_utils::pad_by_offsets(embedded_interaction_data, offsets_with_end,
                                                                 dimension_out_, max_data_per_interaction);

    const auto interaction_type_tokens =
        game_interaction_type_embedding_(flat_tensors.game_interaction_types).unsqueeze(1);
    const auto padded_sequences = torch::cat({interaction_type_tokens, padded_data}, 1);
    const auto valid_token_mask = torch::cat({torch::ones({batch_size, 1}, mask_tensor_options_), data_mask}, 1);

    return attention_utils::masked_self_attention_reduce(game_interaction_attention_, padded_sequences,
                                                         valid_token_mask);
}

torch::Tensor GameInteractionEmbeddingImpl::forward(const std::vector<ProtoBufGameInteraction>& game_interactions,
                                                    torch::Tensor card_indices, torch::Tensor cards) {
    auto flat = flatten_game_interaction_batch(game_interactions);
    auto flat_tensors = flat_game_interaction_batch_to_tensors(flat, device_);
    const auto& ic = flat_tensors.instructions_and_conditions;

    auto embedded_instructions_pair = instruction_embedding_->forward(ic.instructions);
    auto embedded_conditions_pair = condition_embedding_->forward(ic.conditions);
    auto attack_energy_costs = embed_attack_energy_costs(ic);
    const auto embedded_attack_data = embed_attacks(embedded_instructions_pair, ic.instruction_attack_indices,
                                                    attack_energy_costs, ic.energy_slot_per_token);
    const auto embedded_ability_data =
        embed_ability(embedded_instructions_pair, ic.instruction_ability_indices, embedded_conditions_pair,
                      ic.ability_condition_row_for_instruction_ability);

    const auto embedded_target_data = embed_target_data(flat_tensors, card_indices, cards);
    const auto embedded_number_data = embed_number_data(flat_tensors.number_data);
    const auto interaction_card_data = resolve_cards_from_deck_ids(
        flat_tensors.interaction_card_deck_id, card_indices, cards, "embedGameInteraction interaction_card_data");
    const auto embedded_select_from = select_from_embedding_(flat_tensors.select_from);

    const auto embedded_interaction_data =
        embed_interaction_data(flat_tensors, embedded_target_data, embedded_number_data, interaction_card_data,
                               embedded_attack_data, embedded_ability_data, embedded_select_from);
    return reduce_game_interactions(flat_tensors, embedded_interaction_data);
}