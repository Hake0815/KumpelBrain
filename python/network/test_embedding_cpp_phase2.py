import os
import tempfile

import torch

import card_embedding
import embedding_cpp


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


def _leaf(field: int, op: int, value: int) -> dict:
    return {
        "IsLeaf": True,
        "Condition": {"Field": field, "Operation": op, "Value": value},
    }


def _node(logical_operator: int, operands: list[dict]) -> dict:
    return {"IsLeaf": False, "LogicalOperator": logical_operator, "Operands": operands}


def main() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    dim = 32
    n = 12

    py_shared = card_embedding.SharedEmbeddingHolder(dim)
    cpp_shared = embedding_cpp.SharedEmbeddingHolder(dim)

    py_fc = card_embedding.FilterConditionEmbedding(py_shared, dim)
    cpp_fc = embedding_cpp.FilterConditionEmbedding(cpp_shared, dim)
    _with_loaded_weights(py_fc, cpp_fc)

    field = torch.randint(0, 6, (n,), dtype=torch.int64)
    cmp = torch.randint(0, 5, (n,), dtype=torch.int64)
    value = torch.randint(0, 10, (n,), dtype=torch.int64)

    _assert_close(
        "FilterConditionEmbedding",
        py_fc.forward_v2(field, cmp, value),
        cpp_fc.forward_v2(field, cmp, value),
    )

    py_filter = card_embedding.FilterEmbedding(py_shared, dim)
    cpp_filter = embedding_cpp.FilterEmbedding(cpp_shared, dim)
    _with_loaded_weights(py_filter, cpp_filter)

    filter_batch = [
        [
            _leaf(3, 1, 1),
            _node(1, [_leaf(4, 2, 3), _leaf(5, 1, 10)]),
        ],
        [
            _node(
                2,
                [
                    _leaf(3, 1, 2),
                    _node(1, [_leaf(4, 1, 5), _leaf(5, 3, 7)]),
                ],
            )
        ],
    ]

    _assert_close(
        "FilterEmbedding",
        py_filter.forward(filter_batch),
        cpp_filter.forward(filter_batch),
    )
    print("Phase 2 parity passed.")


if __name__ == "__main__":
    main()
