#ifndef SHARED_INSTRUCTION_EMBEDDINGS_FACTORY_H
#define SHARED_INSTRUCTION_EMBEDDINGS_FACTORY_H

#include <memory>

#include <torch/torch.h>

#include "network/include/SharedEmbeddingHolder.h"
#include "network/include/SharedInstructionEmbeddings.h"

SharedInstructionEmbeddings create_shared_instruction_embeddings(
    torch::nn::Module& register_on, std::shared_ptr<SharedEmbeddingHolderImpl> shared, int64_t dimension_out,
    torch::Device device = torch::kCPU, torch::Dtype dtype = torch::kFloat);

#endif
