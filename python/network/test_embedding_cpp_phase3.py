import os
import tempfile

import torch

import card_embedding
import embedding_cpp


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> None:
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


def build_instruction_batch() -> list[list[dict]]:
    return [
        [
            {
                "InstructionType": 1,
                "ConditionType": 1,
                "Data": [
                    {"InstructionDataType": 0, "Payload": {"AttackTarget": 0, "Damage": 4}},
                    {"InstructionDataType": 2, "Payload": {"Amount": {"Min": 1, "Max": 3}, "FromPosition": 2}},
                    {
                        "InstructionDataType": 4,
                        "Payload": {
                            "Filter": [
                                _leaf(3, 1, 2),
                                {"IsLeaf": False, "LogicalOperator": 1, "Operands": [_leaf(4, 2, 3), _leaf(5, 1, 10)]},
                            ]
                        },
                    },
                ],
            },
            {
                "InstructionType": 3,
                "ConditionType": 3,
                "Data": [
                    {"InstructionDataType": 1, "Payload": {"TargetSource": 2}},
                    {"InstructionDataType": 3, "Payload": {"ReturnToDeckType": 1, "FromPosition": 4}},
                    {"InstructionDataType": 5, "Payload": {"PlayerTarget": 1}},
                ],
            },
        ],
        [
            {
                "InstructionType": 2,
                "ConditionType": 2,
                "Data": [
                    {"InstructionDataType": 0, "Payload": {"AttackTarget": 0, "Damage": 2}},
                    {
                        "InstructionDataType": 4,
                        "Payload": {
                            "Filter": [
                                {"IsLeaf": False, "LogicalOperator": 2, "Operands": [_leaf(3, 1, 1), _leaf(4, 1, 5)]}
                            ]
                        },
                    },
                ],
            }
        ],
    ]


def main() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    dim = 32
    batch = build_instruction_batch()

    py_shared = card_embedding.SharedEmbeddingHolder(dim)
    cpp_shared = embedding_cpp.SharedEmbeddingHolder(dim)

    py_instruction = card_embedding.InstructionEmbedding(py_shared, dim)
    cpp_instruction = embedding_cpp.InstructionEmbedding(cpp_shared, dim)
    _with_loaded_weights(py_instruction, cpp_instruction)
    _assert_close("InstructionEmbedding", py_instruction(batch), cpp_instruction.forward(batch))

    py_condition = card_embedding.ConditionEmbedding(py_shared, dim)
    cpp_condition = embedding_cpp.ConditionEmbedding(cpp_shared, dim)
    _with_loaded_weights(py_condition, cpp_condition)
    _assert_close("ConditionEmbedding", py_condition(batch), cpp_condition.forward(batch))

    print("Phase 3 parity passed.")


if __name__ == "__main__":
    main()
