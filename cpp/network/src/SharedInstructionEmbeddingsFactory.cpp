#include "network/include/SharedInstructionEmbeddingsFactory.h"

#include "network/include/AbilityEmbedding.h"
#include "network/include/AttackEmbedding.h"
#include "network/include/ConditionEmbedding.h"
#include "network/include/InstructionDataEmbedding.h"
#include "network/include/InstructionEmbedding.h"

SharedInstructionEmbeddings create_shared_instruction_embeddings(
    torch::nn::Module& register_on, std::shared_ptr<SharedEmbeddingHolderImpl> shared, int64_t dimension_out,
    torch::Device device, torch::Dtype dtype) {
    auto instruction_data_embedding = register_on.register_module(
        "instruction_data_embedding", InstructionDataEmbedding(shared, dimension_out, device, dtype));
    auto instruction_embedding = register_on.register_module(
        "instruction_embedding",
        InstructionEmbedding(instruction_data_embedding, shared, dimension_out, device, dtype));
    auto condition_embedding = register_on.register_module(
        "condition_embedding", ConditionEmbedding(instruction_data_embedding, shared, dimension_out, device, dtype));
    auto attack_embedding =
        register_on.register_module("attack_embedding", AttackEmbedding(dimension_out, device, dtype));
    auto ability_embedding =
        register_on.register_module("ability_embedding", AbilityEmbedding(dimension_out, device, dtype));

    return SharedInstructionEmbeddings{
        instruction_data_embedding,
        instruction_embedding,
        condition_embedding,
        attack_embedding,
        ability_embedding,
    };
}
