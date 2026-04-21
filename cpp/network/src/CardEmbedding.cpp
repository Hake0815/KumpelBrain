#include "../include/CardEmbedding.h"

#include <ATen/ops/cat.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../include/AttentionUtils.h"
#include "../include/SharedConstants.h"

namespace {

std::shared_ptr<std::vector<int64_t>> get_batch_indices_from_map(
    std::unordered_map<std::string, std::shared_ptr<std::vector<int64_t>>>& map, const std::string& key) {
    auto& batch_indices = map[key];
    if (!batch_indices) {
        batch_indices = std::make_shared<std::vector<int64_t>>();
    }
    return batch_indices;
}

template <typename T>
void append_proto_list(std::vector<T>& dst, const google::protobuf::RepeatedPtrField<T>& src) {
    dst.reserve(dst.size() + static_cast<size_t>(src.size()));
    for (const auto& item : src) {
        dst.push_back(item);
    }
}

template <typename T>
void append_values(std::vector<T>& dst, const std::vector<T>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

torch::Tensor sparse_adjacency_from_row_col(std::vector<int64_t>& row_indices, std::vector<int64_t>& col_indices,
                                            int64_t num_cards, torch::Dtype dtype, torch::Device device) {
    torch::Tensor indices_cpu;
    torch::Tensor values_cpu;
    if (row_indices.empty()) {
        indices_cpu = torch::empty({2, 0}, torch::TensorOptions().dtype(torch::kInt64));
        values_cpu = torch::empty({0}, torch::TensorOptions().dtype(dtype));
    } else {
        const int64_t nnz = static_cast<int64_t>(row_indices.size());
        auto rows = torch::from_blob(row_indices.data(), {nnz}, torch::kInt64).clone();
        auto cols = torch::from_blob(col_indices.data(), {nnz}, torch::kInt64).clone();
        indices_cpu = torch::stack({rows, cols}, 0);
        values_cpu = torch::ones({nnz}, torch::TensorOptions().dtype(dtype));
    }
    return torch::sparse_coo_tensor(indices_cpu, values_cpu, {num_cards, num_cards},
                                    torch::TensorOptions().dtype(dtype).device(torch::kCPU))
        .coalesce()
        .to(device);
}

torch::Tensor adjacency_from_ptr_map(
    const std::unordered_map<int64_t, std::vector<std::shared_ptr<int64_t>>>& ptr_map, int64_t num_cards,
    torch::Dtype dtype, torch::Device device) {
    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    size_t reserve = 0;
    for (const auto& entry : ptr_map) {
        reserve += entry.second.size();
    }
    row_indices.reserve(reserve);
    col_indices.reserve(reserve);
    for (const auto& [host_index, index_ptrs] : ptr_map) {
        for (const auto& batch_index_ptr : index_ptrs) {
            if (!batch_index_ptr) {
                continue;
            }
            const int64_t col = *batch_index_ptr;
            if (col >= 0) {
                row_indices.push_back(host_index);
                col_indices.push_back(col);
            }
        }
    }
    return sparse_adjacency_from_row_col(row_indices, col_indices, num_cards, dtype, device);
}

void reserve_card_features(CardFeatures& f, int64_t batch_size) {
    const auto n = static_cast<size_t>(batch_size);
    f.card_type.reserve(n);
    f.card_subtype.reserve(n);
    f.energy_type.reserve(n);
    f.energy_type_context.reserve(n);
    f.energy_type_mask.reserve(n);
    f.max_hp.reserve(n);
    f.max_hp_mask.reserve(n);
    f.weakness.reserve(n);
    f.weakness_mask.reserve(n);
    f.resistance.reserve(n);
    f.resistance_mask.reserve(n);
    f.retreat_cost.reserve(n);
    f.retreat_cost_mask.reserve(n);
    f.number_of_prize_cards_on_knockout.reserve(n);
    f.number_of_prize_cards_on_knockout_mask.reserve(n);
    f.current_damage.reserve(n);
    f.current_damage_mask.reserve(n);
    // Variable-length: heuristic reserve to avoid early log2(n) reallocations.
    const auto nv = n * 4;
    f.flattened_pokemon_turn_traits.reserve(nv);
    f.pokemon_turn_trait_card_indices.reserve(nv);
    f.flattened_provided_energies.reserve(nv);
    f.provided_energy_card_indices.reserve(nv);
    f.flattened_attached_energies.reserve(nv);
    f.attached_energy_card_indices.reserve(nv);
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
        register_module("card_pooling_multi_head_attention",
                        MultiHeadAttention(dimension_out, dimension_out, dimension_out,
                                           std::max<int64_t>(dimension_out_ / 16, 4), 8, 0.0, true, device, dtype));
    card_pooling_query_embedding_ =
        register_module("card_pooling_query_embedding", torch::nn::Embedding(1, dimension_out));
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
    mask_tensor_options_ = torch::TensorOptions().device(device_).dtype(torch::kBool);
    index_tensor_options_ = torch::TensorOptions().device(device_).dtype(torch::kInt64);
    float_tensor_options_ = torch::TensorOptions().device(device_).dtype(dtype);
    ones_1x1_bool_ = torch::ones({1, 1}, mask_tensor_options_);
    to(device, dtype);
}

std::pair<torch::Tensor, AdjacencyMatrices> CardEmbeddingImpl::forward(const std::vector<ProtoBufCard>& card_batch) {
    const int64_t batch_size = static_cast<int64_t>(card_batch.size());
    auto card_features = collect_card_features(card_batch);
    auto staged = stage_features(card_features);
    auto [tokens, mask] = embed_card_features(card_features, staged, batch_size);

    auto self_attended = attention_utils::masked_self_attention(card_self_multi_head_attention_, tokens, mask);

    auto query =
        card_pooling_query_embedding_->weight.view({1, 1, dimension_out_}).expand({batch_size, 1, dimension_out_});
    return {attention_utils::masked_attention_pooling(card_pooling_multi_head_attention_, query, self_attended, mask),
            card_features.adjacency_matrices};
}

void CardEmbeddingImpl::append_card_instructions_and_conditions(const std::vector<ProtoBufCard>& card_batch,
                                                                InstructionsAndConditions& instructions_and_conditions,
                                                                int64_t card_index) {
    const auto& card = card_batch[static_cast<size_t>(card_index)];
    if (card.instructions_size() > 0) {
        instructions_and_conditions.instructions.emplace_back();
        append_proto_list(instructions_and_conditions.instructions.back(), card.instructions());
        instructions_and_conditions.instruction_card_parent_indices.push_back({static_cast<int>(card_index), 0});
        instructions_and_conditions.instruction_card_indices.push_back(
            static_cast<int64_t>(instructions_and_conditions.instructions.size() - 1));
    }
    if (card.conditions_size() > 0) {
        instructions_and_conditions.conditions.emplace_back();
        append_proto_list(instructions_and_conditions.conditions.back(), card.conditions());
        instructions_and_conditions.condition_card_parent_indices.push_back({static_cast<int>(card_index), 0});
        instructions_and_conditions.condition_card_indices.push_back(
            static_cast<int64_t>(instructions_and_conditions.conditions.size() - 1));
    }
    if (card.has_ability()) {
        const auto& ability = card.ability();
        int64_t ability_condition_row = -1;
        if (ability.conditions_size() > 0) {
            instructions_and_conditions.conditions.emplace_back();
            append_proto_list(instructions_and_conditions.conditions.back(), ability.conditions());
            instructions_and_conditions.condition_card_parent_indices.push_back({static_cast<int>(card_index), 0});
            ability_condition_row = static_cast<int64_t>(instructions_and_conditions.conditions.size() - 1);
        }
        if (ability.instructions_size() > 0) {
            instructions_and_conditions.instructions.emplace_back();
            append_proto_list(instructions_and_conditions.instructions.back(), ability.instructions());
            instructions_and_conditions.instruction_card_parent_indices.push_back({static_cast<int>(card_index), 0});
            instructions_and_conditions.instruction_ability_indices.push_back(
                static_cast<int64_t>(instructions_and_conditions.instructions.size() - 1));
            instructions_and_conditions.ability_condition_row_for_instruction_ability.push_back(ability_condition_row);
        }
    }
    if (card.attacks_size() > 0) {
        for (int attack_index = 0; attack_index < card.attacks_size(); ++attack_index) {
            const auto& attack = card.attacks(attack_index);
            if (attack.instructions_size() > 0) {
                const int64_t attack_slot =
                    static_cast<int64_t>(instructions_and_conditions.instruction_attack_indices.size());
                if (attack.energy_cost_size() > 0) {
                    for (const auto energy_type : attack.energy_cost()) {
                        instructions_and_conditions.energy_flat.push_back(static_cast<int64_t>(energy_type));
                        instructions_and_conditions.energy_slot_per_token.push_back(attack_slot);
                    }
                }
                instructions_and_conditions.instructions.emplace_back();
                append_proto_list(instructions_and_conditions.instructions.back(), attack.instructions());
                instructions_and_conditions.instruction_card_parent_indices.push_back(
                    {static_cast<int>(card_index), attack_index});
                instructions_and_conditions.instruction_attack_indices.push_back(
                    static_cast<int64_t>(instructions_and_conditions.instructions.size() - 1));
            }
        }
    }
}

CardFeatures CardEmbeddingImpl::collect_card_features(const std::vector<ProtoBufCard>& card_batch) {
    CardFeatures card_features;
    reserve_card_features(card_features, static_cast<int64_t>(card_batch.size()));
    std::unordered_map<int64_t, std::shared_ptr<std::vector<int64_t>>> evolves_from_matrix;
    std::unordered_map<std::string, std::shared_ptr<std::vector<int64_t>>> name_to_batch_index;
    std::string player_prefix;

    std::unordered_map<int64_t, std::vector<std::shared_ptr<int64_t>>> attached_energy_cards_matrix;
    std::unordered_map<int64_t, std::vector<std::shared_ptr<int64_t>>> pre_evolutions_matrix;
    /// Last batch index seen for each deck_id; assumes at most one card per deck_id in the batch.
    std::unordered_map<int64_t, std::shared_ptr<int64_t>> deck_id_to_card_index;

    for (int64_t card_index = 0; card_index < static_cast<int64_t>(card_batch.size()); ++card_index) {
        const auto& card = card_batch[static_cast<size_t>(card_index)];

        if (card.deck_id() < DECK_SIZE) {
            player_prefix = "player1_";
        } else {
            player_prefix = "player2_";
        }
        auto batch_indices_of_cards_with_same_name =
            get_batch_indices_from_map(name_to_batch_index, player_prefix + card.name());
        batch_indices_of_cards_with_same_name->push_back(card_index);

        auto& ptr = deck_id_to_card_index[card.deck_id()];
        if (!ptr) {
            ptr = std::make_shared<int64_t>(card_index);
        } else {
            *ptr = card_index;
        }

        if (card.has_evolves_from()) {
            const auto& pre_evolution = player_prefix + card.evolves_from();
            auto batch_indices_of_pre_evolution = get_batch_indices_from_map(name_to_batch_index, pre_evolution);
            evolves_from_matrix[card_index] = batch_indices_of_pre_evolution;
        }

        if (card.attached_energy_cards_size() > 0) {
            for (const auto energy_deck_id : card.attached_energy_cards()) {
                auto& batch_index_of_attached_energy_card = deck_id_to_card_index[energy_deck_id];
                if (!batch_index_of_attached_energy_card) {
                    batch_index_of_attached_energy_card = std::make_shared<int64_t>(-1);
                }
                attached_energy_cards_matrix[card_index].push_back(batch_index_of_attached_energy_card);
            }
        }

        if (card.pre_evolution_ids_size() > 0) {
            for (const auto pre_evolution_deck_id : card.pre_evolution_ids()) {
                auto& batch_index_of_pre_evolution = deck_id_to_card_index[pre_evolution_deck_id];
                if (!batch_index_of_pre_evolution) {
                    batch_index_of_pre_evolution = std::make_shared<int64_t>(-1);
                }
                pre_evolutions_matrix[card_index].push_back(batch_index_of_pre_evolution);
            }
        }

        append_card_instructions_and_conditions(card_batch, card_features.instructions_and_conditions, card_index);
        card_features.card_type.push_back(static_cast<int64_t>(card.card_type()));

        card_features.card_subtype.push_back(static_cast<int64_t>(card.card_subtype()));

        card_features.energy_type.push_back(static_cast<int64_t>(card.energy_type()));
        const int64_t energy_type_context =
            card.card_type() == gamecore::serialization::ProtoBufCardType::CARD_TYPE_POKEMON
                ? EnergyTypeContext::POKEMON_TYPE
                : EnergyTypeContext::ENERGY_TYPE;
        card_features.energy_type_context.push_back(energy_type_context);
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

    const int64_t num_cards = static_cast<int64_t>(card_batch.size());
    std::vector<int64_t> evolves_from_row_indices;
    std::vector<int64_t> evolves_from_col_indices;
    evolves_from_row_indices.reserve(evolves_from_matrix.size());
    evolves_from_col_indices.reserve(evolves_from_matrix.size());
    for (const auto& [card_index, pre_evolution_indices] : evolves_from_matrix) {
        for (const int64_t pre_index : *pre_evolution_indices) {
            evolves_from_row_indices.push_back(card_index);
            evolves_from_col_indices.push_back(pre_index);
        }
    }

    card_features.adjacency_matrices.evolves_from_adjacency =
        sparse_adjacency_from_row_col(evolves_from_row_indices, evolves_from_col_indices, num_cards, dtype_, device_);

    card_features.adjacency_matrices.attached_energy_adjacency =
        adjacency_from_ptr_map(attached_energy_cards_matrix, num_cards, dtype_, device_);
    card_features.adjacency_matrices.pre_evolutions_adjacency =
        adjacency_from_ptr_map(pre_evolutions_matrix, num_cards, dtype_, device_);
    return card_features;
}

StagedTensors CardEmbeddingImpl::stage_features(const CardFeatures& f) {
    // Pack all int64 fields into one host buffer, upload once.
    const size_t n_card_type = f.card_type.size();
    const size_t n_card_subtype = f.card_subtype.size();
    const size_t n_max_hp = f.max_hp.size();
    const size_t n_retreat_cost = f.retreat_cost.size();
    const size_t n_nprize = f.number_of_prize_cards_on_knockout.size();
    const size_t n_current_damage = f.current_damage.size();
    const size_t n_traits = f.flattened_pokemon_turn_traits.size();
    const size_t n_trait_idx = f.pokemon_turn_trait_card_indices.size();
    const size_t n_provided_idx = f.provided_energy_card_indices.size();
    const size_t n_attached_idx = f.attached_energy_card_indices.size();
    // Energy-type indices group (order used by embed_energy_type_features narrow).
    const size_t n_et_type = f.energy_type.size();
    const size_t n_et_weak = f.weakness.size();
    const size_t n_et_resist = f.resistance.size();
    const size_t n_et_provided = f.flattened_provided_energies.size();
    const size_t n_et_attached = f.flattened_attached_energies.size();
    const size_t n_et_attack = f.instructions_and_conditions.energy_flat.size();
    const size_t n_et_total = n_et_type + n_et_weak + n_et_resist + n_et_provided + n_et_attached + n_et_attack;
    // Contexts: per-card value for energy_type slot + repeats for the rest.
    const size_t n_ctx_total = n_et_total;

    const size_t total_int64 = n_card_type + n_card_subtype + n_max_hp + n_retreat_cost + n_nprize + n_current_damage +
                               n_traits + n_trait_idx + n_provided_idx + n_attached_idx + n_et_total + n_ctx_total;

    std::vector<int64_t> int64_host;
    int64_host.reserve(total_int64);

    const auto push_block = [&](const std::vector<int64_t>& v) {
        int64_host.insert(int64_host.end(), v.begin(), v.end());
    };
    const auto push_repeat = [&](int64_t value, size_t count) { int64_host.insert(int64_host.end(), count, value); };

    int64_t off = 0;
    const int64_t off_card_type = off;
    push_block(f.card_type);
    off += static_cast<int64_t>(n_card_type);
    const int64_t off_card_subtype = off;
    push_block(f.card_subtype);
    off += static_cast<int64_t>(n_card_subtype);
    const int64_t off_max_hp = off;
    push_block(f.max_hp);
    off += static_cast<int64_t>(n_max_hp);
    const int64_t off_retreat_cost = off;
    push_block(f.retreat_cost);
    off += static_cast<int64_t>(n_retreat_cost);
    const int64_t off_nprize = off;
    push_block(f.number_of_prize_cards_on_knockout);
    off += static_cast<int64_t>(n_nprize);
    const int64_t off_current_damage = off;
    push_block(f.current_damage);
    off += static_cast<int64_t>(n_current_damage);
    const int64_t off_traits = off;
    push_block(f.flattened_pokemon_turn_traits);
    off += static_cast<int64_t>(n_traits);
    const int64_t off_trait_idx = off;
    push_block(f.pokemon_turn_trait_card_indices);
    off += static_cast<int64_t>(n_trait_idx);
    const int64_t off_provided_idx = off;
    push_block(f.provided_energy_card_indices);
    off += static_cast<int64_t>(n_provided_idx);
    const int64_t off_attached_idx = off;
    push_block(f.attached_energy_card_indices);
    off += static_cast<int64_t>(n_attached_idx);

    // Energy-type indices: [energy_type, weakness, resistance, provided, attached, attack_cost]
    const int64_t off_et = off;
    push_block(f.energy_type);
    push_block(f.weakness);
    push_block(f.resistance);
    push_block(f.flattened_provided_energies);
    push_block(f.flattened_attached_energies);
    push_block(f.instructions_and_conditions.energy_flat);
    off += static_cast<int64_t>(n_et_total);

    // Energy-type contexts: per-card for first block, constants for the rest.
    const int64_t off_ctx = off;
    push_block(f.energy_type_context);
    push_repeat(EnergyTypeContext::WEAKNESS, n_et_weak);
    push_repeat(EnergyTypeContext::RESISTANCE, n_et_resist);
    push_repeat(EnergyTypeContext::ENERGY_PROVIDED, n_et_provided);
    push_repeat(EnergyTypeContext::ENERGY_ATTACHED, n_et_attached);
    push_repeat(EnergyTypeContext::ATTACK_COST, n_et_attack);
    off += static_cast<int64_t>(n_ctx_total);

    auto int64_buf = torch::tensor(int64_host, index_tensor_options_);

    // Pack all uint8 mask fields into one host buffer, upload + cast to bool once.
    const size_t n_et_mask = f.energy_type_mask.size();
    const size_t n_weak_mask = f.weakness_mask.size();
    const size_t n_resist_mask = f.resistance_mask.size();
    const size_t n_max_hp_mask = f.max_hp_mask.size();
    const size_t n_retreat_mask = f.retreat_cost_mask.size();
    const size_t n_nprize_mask = f.number_of_prize_cards_on_knockout_mask.size();
    const size_t n_current_damage_mask = f.current_damage_mask.size();
    const size_t total_u8 = n_et_mask + n_weak_mask + n_resist_mask + n_max_hp_mask + n_retreat_mask + n_nprize_mask +
                            n_current_damage_mask;

    std::vector<uint8_t> u8_host;
    u8_host.reserve(total_u8);
    int64_t uoff = 0;
    const int64_t uoff_et = uoff;
    append_values(u8_host, f.energy_type_mask);
    uoff += static_cast<int64_t>(n_et_mask);
    const int64_t uoff_weak = uoff;
    append_values(u8_host, f.weakness_mask);
    uoff += static_cast<int64_t>(n_weak_mask);
    const int64_t uoff_resist = uoff;
    append_values(u8_host, f.resistance_mask);
    uoff += static_cast<int64_t>(n_resist_mask);
    const int64_t uoff_max_hp = uoff;
    append_values(u8_host, f.max_hp_mask);
    uoff += static_cast<int64_t>(n_max_hp_mask);
    const int64_t uoff_retreat = uoff;
    append_values(u8_host, f.retreat_cost_mask);
    uoff += static_cast<int64_t>(n_retreat_mask);
    const int64_t uoff_nprize = uoff;
    append_values(u8_host, f.number_of_prize_cards_on_knockout_mask);
    uoff += static_cast<int64_t>(n_nprize_mask);
    const int64_t uoff_current_damage = uoff;
    append_values(u8_host, f.current_damage_mask);
    uoff += static_cast<int64_t>(n_current_damage_mask);

    auto u8_buf = torch::tensor(u8_host, torch::TensorOptions().device(device_).dtype(torch::kUInt8)).to(torch::kBool);

    StagedTensors s;
    s.card_type = int64_buf.narrow(0, off_card_type, static_cast<int64_t>(n_card_type));
    s.card_subtype = int64_buf.narrow(0, off_card_subtype, static_cast<int64_t>(n_card_subtype));
    s.max_hp = int64_buf.narrow(0, off_max_hp, static_cast<int64_t>(n_max_hp));
    s.retreat_cost = int64_buf.narrow(0, off_retreat_cost, static_cast<int64_t>(n_retreat_cost));
    s.number_of_prize_cards_on_knockout = int64_buf.narrow(0, off_nprize, static_cast<int64_t>(n_nprize));
    s.current_damage = int64_buf.narrow(0, off_current_damage, static_cast<int64_t>(n_current_damage));
    s.flattened_pokemon_turn_traits = int64_buf.narrow(0, off_traits, static_cast<int64_t>(n_traits));
    s.pokemon_turn_trait_card_indices = int64_buf.narrow(0, off_trait_idx, static_cast<int64_t>(n_trait_idx));
    s.provided_energy_card_indices = int64_buf.narrow(0, off_provided_idx, static_cast<int64_t>(n_provided_idx));
    s.attached_energy_card_indices = int64_buf.narrow(0, off_attached_idx, static_cast<int64_t>(n_attached_idx));
    s.energy_type_indices = int64_buf.narrow(0, off_et, static_cast<int64_t>(n_et_total));
    s.energy_type_contexts = int64_buf.narrow(0, off_ctx, static_cast<int64_t>(n_ctx_total));

    s.energy_type_mask = u8_buf.narrow(0, uoff_et, static_cast<int64_t>(n_et_mask));
    s.weakness_mask = u8_buf.narrow(0, uoff_weak, static_cast<int64_t>(n_weak_mask));
    s.resistance_mask = u8_buf.narrow(0, uoff_resist, static_cast<int64_t>(n_resist_mask));
    s.max_hp_mask = u8_buf.narrow(0, uoff_max_hp, static_cast<int64_t>(n_max_hp_mask));
    s.retreat_cost_mask = u8_buf.narrow(0, uoff_retreat, static_cast<int64_t>(n_retreat_mask));
    s.number_of_prize_cards_on_knockout_mask = u8_buf.narrow(0, uoff_nprize, static_cast<int64_t>(n_nprize_mask));
    s.current_damage_mask = u8_buf.narrow(0, uoff_current_damage, static_cast<int64_t>(n_current_damage_mask));
    return s;
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_features(const CardFeatures& card_features,
                                                                               const StagedTensors& staged,
                                                                               int64_t batch_size) {
    auto [embedded_energy_type_tokens, embedded_energy_type_mask, attack_energy_costs] =
        embed_energy_type_features(card_features, staged, batch_size);
    auto [instructions_and_conditions_tokens, instructions_and_conditions_mask] =
        embed_instructions_and_conditions(card_features.instructions_and_conditions, attack_energy_costs, batch_size);

    auto card_type_mask = ones_1x1_bool_.expand({batch_size, 1});
    auto card_type_tokens = shared_embedding_holder_->card_type_embedding_(staged.card_type).unsqueeze(1);

    auto card_subtype_mask = ones_1x1_bool_.expand({batch_size, 1});
    auto card_subtype_tokens = shared_embedding_holder_->card_subtype_embedding_(staged.card_subtype).unsqueeze(1);

    auto max_hp_mask = staged.max_hp_mask.unsqueeze(1);
    auto max_hp_tokens =
        (shared_embedding_holder_->hp_embedding_(staged.max_hp.unsqueeze(-1)) * max_hp_mask).unsqueeze(1);

    auto retreat_cost_mask = staged.retreat_cost_mask.unsqueeze(1);
    auto retreat_cost_tokens =
        (retreat_cost_embedding_(staged.retreat_cost.unsqueeze(-1)) * retreat_cost_mask).unsqueeze(1);

    auto number_of_prize_cards_on_knockout_mask = staged.number_of_prize_cards_on_knockout_mask.unsqueeze(1);
    auto number_of_prize_cards_on_knockout_tokens =
        (number_of_prize_cards_on_knockout_embedding_(staged.number_of_prize_cards_on_knockout.unsqueeze(-1)) *
         number_of_prize_cards_on_knockout_mask)
            .unsqueeze(1);

    auto current_damage_mask = staged.current_damage_mask.unsqueeze(1);
    auto current_damage_tokens =
        (current_damage_embedding_(staged.current_damage.unsqueeze(-1)) * current_damage_mask).unsqueeze(1);

    auto [pokemon_turn_trait_tokens, pokemon_turn_trait_mask] = combine_flat_embedded_card_feature(
        pokemon_turn_trait_embedding_->forward(staged.flattened_pokemon_turn_traits),
        card_features.pokemon_turn_trait_card_indices, staged.pokemon_turn_trait_card_indices, batch_size);

    return {torch::cat({card_type_tokens, card_subtype_tokens, embedded_energy_type_tokens, max_hp_tokens,
                        retreat_cost_tokens, number_of_prize_cards_on_knockout_tokens, current_damage_tokens,
                        pokemon_turn_trait_tokens, instructions_and_conditions_tokens},
                       1),
            torch::cat({card_type_mask, card_subtype_mask, embedded_energy_type_mask, max_hp_mask, retreat_cost_mask,
                        number_of_prize_cards_on_knockout_mask, current_damage_mask, pokemon_turn_trait_mask,
                        instructions_and_conditions_mask},
                       1)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_energy_type_features(
    const CardFeatures& card_features, const StagedTensors& staged, int64_t batch_size) {
    auto energy_type_mask = staged.energy_type_mask.unsqueeze(1);
    auto weakness_mask = staged.weakness_mask.unsqueeze(1);
    auto resistance_mask = staged.resistance_mask.unsqueeze(1);

    auto embedded_energy_types = shared_embedding_holder_->energy_type_embedding_->forward(staged.energy_type_indices,
                                                                                           staged.energy_type_contexts);

    int64_t current_offset = 0;

    auto energy_type_tokens =
        (embedded_energy_types.narrow(0, current_offset, card_features.energy_type.size()) * energy_type_mask)
            .unsqueeze(1);
    current_offset += card_features.energy_type.size();

    auto weakness_tokens =
        (embedded_energy_types.narrow(0, current_offset, card_features.weakness.size()) * weakness_mask).unsqueeze(1);
    current_offset += card_features.weakness.size();

    auto resistance_tokens =
        (embedded_energy_types.narrow(0, current_offset, card_features.resistance.size()) * resistance_mask)
            .unsqueeze(1);
    current_offset += card_features.resistance.size();

    auto embedded_provided_energies =
        embedded_energy_types.narrow(0, current_offset, card_features.flattened_provided_energies.size());
    current_offset += card_features.flattened_provided_energies.size();

    auto embedded_attached_energies =
        embedded_energy_types.narrow(0, current_offset, card_features.flattened_attached_energies.size());
    current_offset += card_features.flattened_attached_energies.size();

    auto attack_energy_costs =
        embedded_energy_types.narrow(0, current_offset, card_features.instructions_and_conditions.energy_flat.size());

    auto [provided_energy_tokens, provided_energy_mask] =
        combine_flat_embedded_card_feature(embedded_provided_energies, card_features.provided_energy_card_indices,
                                           staged.provided_energy_card_indices, batch_size);
    auto [attached_energy_tokens, attached_energy_mask] =
        combine_flat_embedded_card_feature(embedded_attached_energies, card_features.attached_energy_card_indices,
                                           staged.attached_energy_card_indices, batch_size);

    return {
        torch::cat(
            {energy_type_tokens, weakness_tokens, resistance_tokens, provided_energy_tokens, attached_energy_tokens},
            1),
        torch::cat({energy_type_mask, weakness_mask, resistance_mask, provided_energy_mask, attached_energy_mask}, 1),
        attack_energy_costs};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::combine_flat_embedded_card_feature(
    const torch::Tensor& flat_embedded_feature, const std::vector<int64_t>& card_indices,
    const torch::Tensor& card_indices_tensor, int64_t batch_size) {
    if (flat_embedded_feature.size(0) == 0) {
        TORCH_CHECK(
            card_indices.empty(),
            "combine_flat_embedded_card_feature: card_indices must be empty when flat_embedded_feature is empty");
        return {torch::zeros({batch_size, 0, dimension_out_}, float_tensor_options_),
                torch::zeros({batch_size, 0}, mask_tensor_options_)};
    }

    TORCH_CHECK(flat_embedded_feature.size(0) == static_cast<int64_t>(card_indices.size()),
                "combine_flat_embedded_card_feature: flat_embedded_feature and card_indices must have the same length");

    std::vector<int> next_slot(static_cast<size_t>(batch_size), 0);
    const int64_t total_number_of_features = static_cast<int64_t>(card_indices.size());

    std::vector<int64_t> scatter_seq;
    scatter_seq.reserve(static_cast<size_t>(total_number_of_features));
    int max_card_index_repetition = 0;
    for (int64_t card_index : card_indices) {
        const int64_t sequence_index = next_slot[static_cast<size_t>(card_index)]++;
        scatter_seq.push_back(sequence_index);
        max_card_index_repetition = std::max(max_card_index_repetition, next_slot[static_cast<size_t>(card_index)]);
    }

    auto out = torch::zeros({batch_size, max_card_index_repetition, dimension_out_}, float_tensor_options_);
    auto mask = torch::zeros({batch_size, max_card_index_repetition}, mask_tensor_options_);
    auto scatter_seq_tensor = torch::tensor(scatter_seq, index_tensor_options_);
    out.index_put_({card_indices_tensor, scatter_seq_tensor}, flat_embedded_feature);
    mask.index_put_({card_indices_tensor, scatter_seq_tensor}, torch::tensor(true, mask_tensor_options_));
    return {out, mask};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_instructions_and_conditions(
    const InstructionsAndConditions& instructions_and_conditions, const torch::Tensor& attack_energy_costs,
    int64_t batch_size) {
    auto embedded_instructions_pair = instruction_embedding_->forward(instructions_and_conditions.instructions);
    auto embedded_conditions_pair = condition_embedding_->forward(instructions_and_conditions.conditions);

    auto [embedded_attacks, mask_attacks] =
        embed_attacks(embedded_instructions_pair, instructions_and_conditions.instruction_attack_indices,
                      attack_energy_costs, instructions_and_conditions.energy_slot_per_token,
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
    const std::vector<int64_t>& instruction_attack_indices, const torch::Tensor& attack_energy_costs,
    const std::vector<int64_t>& energy_slot_per_token, const std::vector<ParentIndex>& instruction_card_parent_indices,
    int64_t batch_size) {
    if (instruction_attack_indices.empty()) {
        return {torch::zeros({batch_size, 0, dimension_out_}, float_tensor_options_),
                torch::zeros({batch_size, 0}, mask_tensor_options_)};
    }

    /// Build attack_energy_sums: Row i matches instruction_attack_indices[i]: sum of energy_type embeddings for that
    /// attack (zeros if no costs).
    const int64_t num_attacks = static_cast<int64_t>(instruction_attack_indices.size());
    auto attack_energy_sums = torch::zeros({num_attacks, dimension_out_}, float_tensor_options_);
    if (attack_energy_costs.size(0) > 0) {
        auto slot_idx = torch::tensor(energy_slot_per_token, index_tensor_options_);
        attack_energy_sums.index_add_(0, slot_idx, attack_energy_costs);
    }

    const auto attack_instruction_rows = torch::tensor(instruction_attack_indices, index_tensor_options_);
    auto embedded_instruction_attacks = embedded_instructions_pair.first.index_select(0, attack_instruction_rows);
    auto embedded_instruction_attacks_mask = embedded_instructions_pair.second.index_select(0, attack_instruction_rows);
    auto embedded_attacks =
        attack_embedding_->forward(attack_energy_sums, embedded_instruction_attacks, embedded_instruction_attacks_mask);

    int64_t max_attacks = 0;
    std::vector<int64_t> scatter_card;
    std::vector<int64_t> scatter_attack_pos;
    scatter_card.reserve(static_cast<size_t>(num_attacks));
    scatter_attack_pos.reserve(static_cast<size_t>(num_attacks));
    for (int64_t row : instruction_attack_indices) {
        const auto& parent = instruction_card_parent_indices[static_cast<size_t>(row)];
        scatter_card.push_back(static_cast<int64_t>(parent.card));
        scatter_attack_pos.push_back(static_cast<int64_t>(parent.slot));
        max_attacks = std::max(max_attacks, static_cast<int64_t>(parent.slot) + 1);
    }

    auto out = torch::zeros({batch_size, max_attacks, dimension_out_}, float_tensor_options_);
    auto mask = torch::zeros({batch_size, max_attacks}, mask_tensor_options_);
    auto scatter_card_t = torch::tensor(scatter_card, index_tensor_options_);
    auto scatter_pos_t = torch::tensor(scatter_attack_pos, index_tensor_options_);
    out.index_put_({scatter_card_t, scatter_pos_t}, embedded_attacks);
    mask.index_put_({scatter_card_t, scatter_pos_t}, torch::tensor(true, mask_tensor_options_));
    return {out, mask};
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_ability(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_ability_indices,
    const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
    const std::vector<int64_t>& ability_condition_row_for_instruction_ability,
    const std::vector<ParentIndex>& instruction_card_parent_indices, int64_t batch_size) {
    if (instruction_ability_indices.empty()) {
        auto empty_slot = torch::zeros({batch_size, 1, dimension_out_}, float_tensor_options_);
        auto empty_mask = torch::zeros({batch_size, 1}, mask_tensor_options_);
        return {empty_slot, empty_mask};
    }

    auto instruction_index_tensor = torch::tensor(instruction_ability_indices, index_tensor_options_);
    auto embedded_instruction_abilities = embedded_instructions_pair.first.index_select(0, instruction_index_tensor);
    auto embedded_instruction_abilities_mask =
        embedded_instructions_pair.second.index_select(0, instruction_index_tensor);

    const int64_t number_of_abilities = static_cast<int64_t>(instruction_ability_indices.size());

    // All condition groups share the same padded sequence length (dim 1 of embedded_conditions_pair).
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

    auto embedded_abilities = ability_embedding_->forward(embedded_instruction_abilities,
                                                          embedded_instruction_abilities_mask, cond_vals, cond_mask);

    return pad_to_batch(instruction_ability_indices, instruction_card_parent_indices, batch_size, embedded_abilities);
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_instructions(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_instructions_pair,
    const std::vector<int64_t>& instruction_card_indices,
    const std::vector<ParentIndex>& instruction_card_parent_indices, int64_t batch_size) {
    if (instruction_card_indices.empty()) {
        auto empty_slot = torch::zeros({batch_size, 1, dimension_out_}, float_tensor_options_);
        auto empty_mask = torch::zeros({batch_size, 1}, mask_tensor_options_);
        return {empty_slot, empty_mask};
    }
    auto card_instruction_rows = torch::tensor(instruction_card_indices, index_tensor_options_);
    auto embedded_instructions = embedded_instructions_pair.first.index_select(0, card_instruction_rows);
    auto embedded_instructions_mask = embedded_instructions_pair.second.index_select(0, card_instruction_rows);

    const int64_t n = embedded_instructions.size(0);
    auto instruction_query =
        card_instruction_query_embedding_->weight.view({1, 1, dimension_out_}).expand({n, 1, dimension_out_});

    auto pooled_instructions = attention_utils::masked_attention_pooling(
        card_instructions_multi_head_attention_, instruction_query, embedded_instructions, embedded_instructions_mask);

    return pad_to_batch(instruction_card_indices, instruction_card_parent_indices, batch_size, pooled_instructions);
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::embed_card_conditions(
    const std::pair<torch::Tensor, torch::Tensor>& embedded_conditions_pair,
    const std::vector<int64_t>& condition_card_indices, const std::vector<ParentIndex>& condition_card_parent_indices,
    int64_t batch_size) {
    if (condition_card_indices.empty()) {
        auto empty_slot = torch::zeros({batch_size, 1, dimension_out_}, float_tensor_options_);
        auto empty_mask = torch::zeros({batch_size, 1}, mask_tensor_options_);
        return {empty_slot, empty_mask};
    }
    auto card_condition_rows = torch::tensor(condition_card_indices, index_tensor_options_);
    auto embedded_conditions = embedded_conditions_pair.first.index_select(0, card_condition_rows);
    auto embedded_conditions_mask = embedded_conditions_pair.second.index_select(0, card_condition_rows);

    const int64_t n = embedded_conditions.size(0);
    auto condition_query =
        card_condition_query_embedding_->weight.view({1, 1, dimension_out_}).expand({n, 1, dimension_out_});

    auto pooled_conditions = attention_utils::masked_attention_pooling(
        card_conditions_multi_head_attention_, condition_query, embedded_conditions, embedded_conditions_mask);

    return pad_to_batch(condition_card_indices, condition_card_parent_indices, batch_size, pooled_conditions);
}

std::pair<torch::Tensor, torch::Tensor> CardEmbeddingImpl::pad_to_batch(
    const std::vector<int64_t>& card_indices, const std::vector<ParentIndex>& card_parent_indices, int64_t batch_size,
    const torch::Tensor& pooled_tokens) {
    auto out = torch::zeros({batch_size, dimension_out_}, float_tensor_options_);
    auto mask = torch::zeros({batch_size}, mask_tensor_options_);

    const auto n = static_cast<int64_t>(card_indices.size());
    std::vector<int64_t> batch_rows;
    batch_rows.reserve(static_cast<size_t>(n));
    for (int64_t idx : card_indices) {
        batch_rows.push_back(static_cast<int64_t>(card_parent_indices[static_cast<size_t>(idx)].card));
    }
    auto batch_rows_t = torch::tensor(batch_rows, index_tensor_options_);
    out.index_put_({batch_rows_t}, pooled_tokens);
    mask.index_fill_(0, batch_rows_t, true);
    return {out.unsqueeze(1), mask.unsqueeze(1)};
}
