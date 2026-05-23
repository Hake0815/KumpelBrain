#ifndef SHARED_INSTRUCTION_EMBEDDINGS_H
#define SHARED_INSTRUCTION_EMBEDDINGS_H

#include "network/include/AbilityEmbedding.h"
#include "network/include/AttackEmbedding.h"
#include "network/include/ConditionEmbedding.h"
#include "network/include/InstructionDataEmbedding.h"
#include "network/include/InstructionEmbedding.h"

struct SharedInstructionEmbeddings {
    InstructionDataEmbedding instruction_data_embedding{nullptr};
    InstructionEmbedding instruction_embedding{nullptr};
    ConditionEmbedding condition_embedding{nullptr};
    AttackEmbedding attack_embedding{nullptr};
    AbilityEmbedding ability_embedding{nullptr};
};

#endif
