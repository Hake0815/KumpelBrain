#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H

#include <cstdint>

#include "network/src/serialization/gamecore_serialization.pb.h"

constexpr int64_t DECK_SIZE = 60;
constexpr int64_t NUMBER_CARD_TYPES = gamecore::serialization::ProtoBufCardType_ARRAYSIZE;
constexpr int64_t NUMBER_CARD_SUBTYPES = gamecore::serialization::ProtoBufCardSubtype_ARRAYSIZE;
constexpr int64_t NUMBER_CARD_POSITIONS = gamecore::serialization::ProtoBufCardPosition_ARRAYSIZE;
constexpr int64_t NUMBER_PLAYER_TARGETS = gamecore::serialization::ProtoBufPlayerTarget_ARRAYSIZE;
constexpr int64_t NUMBER_POKEMON_TURN_TRAITS = gamecore::serialization::ProtoBufPokemonTurnTrait_ARRAYSIZE;
constexpr int64_t NUMBER_OWNERS = gamecore::serialization::ProtoBufOwner_ARRAYSIZE;
constexpr int64_t NUMBER_POSITION_KNOWLEDGE = gamecore::serialization::ProtoBufPositionKnowledge_ARRAYSIZE;
constexpr int64_t NUMBER_CONDITION_TYPES = gamecore::serialization::ProtoBufConditionType_ARRAYSIZE;
constexpr int64_t NUMBER_TARGET_SOURCES = gamecore::serialization::ProtoBufTargetSource_ARRAYSIZE;
constexpr int64_t NUMBER_ENERGY_TYPES = gamecore::serialization::ProtoBufEnergyType_ARRAYSIZE;
constexpr int64_t NUMBER_FILTER_FIELD_TYPES = gamecore::serialization::ProtoBufFilterType_ARRAYSIZE;
constexpr int64_t NUMBER_FILTER_OPERATIONS = gamecore::serialization::ProtoBufFilterOperation_ARRAYSIZE;
constexpr int64_t NUMBER_FILTER_LOGICAL_OPERATORS = gamecore::serialization::ProtoBufFilterLogicalOperator_ARRAYSIZE;
constexpr int64_t NUMBER_INSTRUCTION_TYPES = gamecore::serialization::ProtoBufInstructionType_ARRAYSIZE;
constexpr int64_t NUMBER_INSTRUCTION_DATA_TYPES = gamecore::serialization::ProtoBufInstructionDataType_ARRAYSIZE;
constexpr int64_t NUMBER_RETURN_TO_DECK_TYPES = gamecore::serialization::ProtoBufReturnToDeckType_ARRAYSIZE;

#endif