#include "network/include/DeckIdResolution.h"

#include <sstream>
#include <stdexcept>

namespace {

void throw_invalid_deck_id(int64_t deck_id, const char* context, const char* reason) {
    std::ostringstream message;
    message << context << ": invalid deck_id " << deck_id << " (" << reason << ")";
    throw std::invalid_argument(message.str());
}

}  // namespace

torch::Tensor resolve_cards_from_deck_ids(const torch::Tensor& deck_ids, const torch::Tensor& card_indices,
                                          const torch::Tensor& cards, const char* context) {
    if (deck_ids.numel() == 0) {
        return torch::empty({0, cards.size(1)}, cards.options());
    }
    if (!deck_ids.defined() || deck_ids.scalar_type() != torch::kLong) {
        throw std::invalid_argument(std::string(context) + ": deck_ids must be a defined int64 tensor");
    }
    if (deck_ids.device() != card_indices.device() || deck_ids.device() != cards.device()) {
        throw std::invalid_argument(std::string(context) + ": deck_ids, card_indices, and cards must share device");
    }

    const auto lookup_size = card_indices.size(0);
    const auto deck_ids_cpu = deck_ids.cpu();
    const auto card_indices_cpu = card_indices.cpu();
    for (int64_t i = 0; i < deck_ids_cpu.size(0); ++i) {
        const auto deck_id = deck_ids_cpu[i].item<int64_t>();
        if (deck_id < 0) {
            throw_invalid_deck_id(deck_id, context, "negative deck_id");
        }
        if (deck_id >= lookup_size) {
            throw_invalid_deck_id(deck_id, context, "deck_id out of card_indices range");
        }
        const auto row_index = card_indices_cpu[deck_id].item<int64_t>();
        if (row_index < 0) {
            throw_invalid_deck_id(deck_id, context, "no card with this deck_id in game state");
        }
    }

    const auto row_indices = card_indices.index_select(0, deck_ids);
    return cards.index_select(0, row_indices);
}
