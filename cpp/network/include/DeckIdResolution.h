#ifndef DECK_ID_RESOLUTION_H
#define DECK_ID_RESOLUTION_H

#include <torch/torch.h>

#include <string>

/// Resolves deck ids through \p card_indices and gathers rows from \p cards.
/// Throws std::invalid_argument when a deck id is out of range or absent (-1 in lookup).
torch::Tensor resolve_cards_from_deck_ids(const torch::Tensor& deck_ids, const torch::Tensor& card_indices,
                                          const torch::Tensor& cards, const char* context);

#endif
