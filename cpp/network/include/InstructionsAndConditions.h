#ifndef INSTRUCTIONS_AND_CONDITIONS_H
#define INSTRUCTIONS_AND_CONDITIONS_H

#include <cstdint>
#include <vector>

#include "network/src/serialization/gamecore_serialization.pb.h"

struct ParentIndex {
    int card;
    int slot;
};

struct InstructionsAndConditions {
    std::vector<std::vector<gamecore::serialization::ProtoBufInstruction>> instructions;
    std::vector<std::vector<gamecore::serialization::ProtoBufCondition>> conditions;
    std::vector<ParentIndex> instruction_card_parent_indices;
    std::vector<ParentIndex> condition_card_parent_indices;
    std::vector<int64_t> instruction_card_indices;
    std::vector<int64_t> instruction_ability_indices;
    std::vector<int64_t> instruction_attack_indices;
    std::vector<int64_t> condition_card_indices;
    /// Same length as instruction_ability_indices: global condition row index for that ability's
    /// instructions, or -1 if the ability has no conditions.
    std::vector<int64_t> ability_condition_row_for_instruction_ability;
    std::vector<int64_t> energy_flat;
    std::vector<int64_t> energy_slot_per_token;
};

#endif
