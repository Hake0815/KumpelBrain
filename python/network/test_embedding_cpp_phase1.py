import os
import sys
import tempfile

import torch

import card_embedding

sys.path.insert(0, "../../cpp/build")
import kumpel_embedding


def _assert_close(
    name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5
) -> None:
    if not torch.allclose(a, b, atol=atol):
        raise AssertionError(f"{name} mismatch:\npython={a}\ncpp={b}")


def _with_loaded_weights(py_model, cpp_model):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        path = fp.name
    try:
        py_model.save_weights(path)
        cpp_model.load_weights(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_normalized_linear(dim, batch):
    py_norm = card_embedding.NormalizedLinear(3, dim)
    cpp_norm = kumpel_embedding.NormalizedLinear(3, dim)

    _with_loaded_weights(py_norm, cpp_norm)
    x = torch.randn(batch, 3)
    _assert_close("NormalizedLinear", py_norm(x), cpp_norm.forward(x))


def main() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    dim = 32
    batch = 8

    test_normalized_linear(dim, batch)

    py_shared = card_embedding.SharedEmbeddingHolder(dim)
    cpp_shared = kumpel_embedding.SharedEmbeddingHolder(dim)
    py_shared.eval()
    cpp_shared.eval()

    py_attack = card_embedding.AttackDataEmbedding(dim)
    cpp_attack = kumpel_embedding.AttackDataEmbedding(dim)
    _with_loaded_weights(py_attack, cpp_attack)
    attack_data = torch.randint(0, 1, (batch, 2), dtype=torch.int64)
    attack_data[:, 1] = torch.randint(0, 8, (batch,), dtype=torch.int64)
    _assert_close(
        "AttackDataEmbedding", py_attack(attack_data), cpp_attack.forward(attack_data)
    )

    py_discard = card_embedding.DiscardDataEmbedding(dim)
    cpp_discard = kumpel_embedding.DiscardDataEmbedding(dim)
    _with_loaded_weights(py_discard, cpp_discard)
    discard_data = torch.randint(0, 3, (batch,), dtype=torch.int64)
    _assert_close(
        "DiscardDataEmbedding",
        py_discard(discard_data),
        cpp_discard.forward(discard_data),
    )

    py_amount = card_embedding.CardAmountDataEmbedding(py_shared, dim)
    cpp_amount = kumpel_embedding.CardAmountDataEmbedding(cpp_shared, dim)
    _with_loaded_weights(py_amount, cpp_amount)
    amount_data = torch.randint(0, 8, (batch, 3), dtype=torch.int64)
    amount_data[:, 2] = torch.randint(0, 11, (batch,), dtype=torch.int64)
    _assert_close(
        "CardAmountDataEmbedding",
        py_amount(amount_data),
        cpp_amount.forward(amount_data),
    )

    py_ret = card_embedding.ReturnToDeckTypeDataEmbedding(py_shared, dim)
    cpp_ret = kumpel_embedding.ReturnToDeckTypeDataEmbedding(cpp_shared, dim)
    _with_loaded_weights(py_ret, cpp_ret)
    ret_data = torch.randint(0, 2, (batch, 2), dtype=torch.int64)
    ret_data[:, 1] = torch.randint(0, 11, (batch,), dtype=torch.int64)
    _assert_close(
        "ReturnToDeckTypeDataEmbedding",
        py_ret(ret_data),
        cpp_ret.forward(ret_data),
    )

    py_player = card_embedding.PlayerTargetDataEmbedding(py_shared, dim)
    cpp_player = kumpel_embedding.PlayerTargetDataEmbedding(cpp_shared, dim)
    _with_loaded_weights(py_player, cpp_player)
    player_data = torch.randint(0, 2, (batch,), dtype=torch.int64)
    _assert_close(
        "PlayerTargetDataEmbedding",
        py_player(player_data),
        cpp_player.forward(player_data),
    )

    print("Phase 1 parity passed.")


if __name__ == "__main__":
    main()
