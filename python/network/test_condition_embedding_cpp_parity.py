import os
from pathlib import Path
import sys
import tempfile

import torch

import card_embedding
import instruction_test_data
import nesting
import proto_serialization

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpp" / "build"))
import kumpel_embedding


def _assert_close(
    name: str, a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5
) -> None:
    if not torch.allclose(a, b, atol=atol):
        diff = (a - b).abs()
        max_diff = diff.max().item()
        bad = (diff > atol).nonzero()
        first_bad = tuple(bad[0].tolist()) if bad.numel() > 0 else None
        raise AssertionError(
            f"{name} mismatch: max_diff={max_diff:.6f}, first_bad_index={first_bad}"
        )


def _with_loaded_weights(py_model, cpp_model) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        path = fp.name
    try:
        py_model.save_weights(path)
        cpp_model.load_weights(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def _python_condition_forward_reference(
    py_condition: card_embedding.ConditionEmbedding,
    conditions_batch: list[list[dict]],
) -> torch.Tensor:
    batch_size = len(conditions_batch)
    (
        condition_types,
        condition_indices,
        instruction_data_types,
        instruction_data_type_indices,
        instruction_data,
        instruction_data_indices,
    ) = nesting.flatten_instructions(
        "ConditionType",
        conditions_batch,
        device=py_condition.factory_kwargs["device"],
        dtype=py_condition.factory_kwargs["dtype"],
    )

    condition_type_embeddings = py_condition.condition_type_embedding(condition_types)
    data_tensors = py_condition.instruction_data_embedding(
        condition_indices,
        instruction_data_types,
        instruction_data_type_indices,
        instruction_data,
        instruction_data_indices,
        batch_size,
    )

    condition_embeddings = []
    for i, condition_index in enumerate(condition_indices):
        per_condition_data = data_tensors[
            (instruction_data_type_indices[:, 0:2] == condition_index).sum(1) == 2
        ]
        query = torch.cat(
            [condition_type_embeddings[i].unsqueeze(0), per_condition_data], dim=0
        ).unsqueeze(0)
        condition_embeddings.append(
            (query + py_condition.data_multi_head_attention(query, query, query))
            .sum(1)
            .squeeze(0)
        )

    if condition_embeddings:
        condition_embeddings = torch.stack(condition_embeddings, dim=0)
    else:
        condition_embeddings = torch.empty(
            (0, py_condition.dimension_out),
            device=py_condition.factory_kwargs["device"],
        )

    batched_conditions = []
    for batch_index in range(batch_size):
        per_batch = condition_embeddings[condition_indices[:, 0] == batch_index]
        if per_batch.shape[0] == 0:
            batched_conditions.append(
                torch.zeros(
                    py_condition.dimension_out,
                    device=py_condition.factory_kwargs["device"],
                    dtype=py_condition.factory_kwargs["dtype"],
                )
            )
            continue
        positioned = py_condition._position_embedding(per_batch.unsqueeze(0)).squeeze(0)
        query = positioned.unsqueeze(0)
        batched_conditions.append(
            (query + py_condition.conditions_multi_head_attention(query, query, query))
            .sum(1)
            .squeeze(0)
        )

    return torch.stack(batched_conditions, dim=0)


def main() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    dim = 32
    device = torch.device("cpu")
    conditions_batch = instruction_test_data.conditions_batch
    serialized_conditions_batch = proto_serialization.serialize_condition_batches(
        conditions_batch
    )

    py_shared = card_embedding.SharedEmbeddingHolder(dim, device=device)
    cpp_shared = kumpel_embedding.SharedEmbeddingHolder(dim, device=device)
    _with_loaded_weights(py_shared, cpp_shared)
    py_shared.eval()
    cpp_shared.eval()

    py_instruction_data = card_embedding.InstructionDataEmbedding(
        py_shared, dim, device=device
    )
    cpp_instruction_data = kumpel_embedding.InstructionDataEmbedding(
        cpp_shared, dim, device=device
    )
    _with_loaded_weights(py_instruction_data, cpp_instruction_data)

    py_condition = card_embedding.ConditionEmbedding(
        py_instruction_data, py_shared, dim, device=device
    )
    cpp_condition = kumpel_embedding.ConditionEmbedding(
        cpp_instruction_data, cpp_shared, dim, device=device
    )
    _with_loaded_weights(py_condition, cpp_condition)

    py_out = _python_condition_forward_reference(py_condition, conditions_batch)
    cpp_out = cpp_condition.forward(serialized_conditions_batch)
    _assert_close("ConditionEmbedding", py_out, cpp_out)

    print("ConditionEmbedding parity passed.")


if __name__ == "__main__":
    main()
