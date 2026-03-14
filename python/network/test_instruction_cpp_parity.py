import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable

import torch

import instruction_test_data
import card_embedding
import nesting

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


def _set_constant_parameters(model, value: float = 0.01) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def _benchmark_forward(
    forward_fn: Callable[[], torch.Tensor], warmup_runs: int = 20, runs: int = 200
) -> float:
    for _ in range(warmup_runs):
        forward_fn()

    start = time.perf_counter()
    for _ in range(runs):
        forward_fn()
    total_seconds = time.perf_counter() - start
    return total_seconds / runs


def test_instruction_data_embedding_parity(
    py_shared: card_embedding.SharedEmbeddingHolder,
    cpp_shared: kumpel_embedding.SharedEmbeddingHolder,
    dim: int,
    device: torch.device,
) -> tuple[
    card_embedding.InstructionDataEmbedding, kumpel_embedding.InstructionDataEmbedding
]:
    instructions_batch = instruction_test_data.instructions_batch
    batch_size = len(instructions_batch)

    py_instruction_data = card_embedding.InstructionDataEmbedding(
        py_shared, dim, device=device
    )
    cpp_instruction_data = kumpel_embedding.InstructionDataEmbedding(
        cpp_shared, dim, device=device
    )
    _with_loaded_weights(py_instruction_data, cpp_instruction_data)

    (
        _instruction_types,
        instruction_indices,
        instruction_data_types,
        instruction_data_type_indices,
        instruction_data,
        instruction_data_indices,
    ) = nesting.flatten_instructions(
        "InstructionType", instructions_batch, device=device
    )

    py_out = py_instruction_data(
        instruction_indices,
        instruction_data_types,
        instruction_data_type_indices,
        instruction_data,
        instruction_data_indices,
        batch_size,
    )
    cpp_out = cpp_instruction_data.forward(
        instruction_indices,
        instruction_data_types,
        instruction_data_type_indices,
        instruction_data,
        instruction_data_indices,
        batch_size,
    )
    _assert_close("InstructionDataEmbedding", py_out, cpp_out)

    return py_instruction_data, cpp_instruction_data


def _python_instruction_forward_reference(
    py_instruction: card_embedding.InstructionEmbedding,
    instructions_batch: list[list[dict]],
) -> torch.Tensor:
    batch_size = len(instructions_batch)
    (
        instruction_types,
        instruction_indices,
        instruction_data_types,
        instruction_data_type_indices,
        instruction_data,
        instruction_data_indices,
    ) = nesting.flatten_instructions(
        "InstructionType",
        instructions_batch,
        device=py_instruction.factory_kwargs["device"],
        dtype=py_instruction.factory_kwargs["dtype"],
    )

    instruction_type_embeddings = py_instruction.instruction_type_embedding(
        instruction_types
    )
    data_tensors = py_instruction.instruction_data_embedding(
        instruction_indices,
        instruction_data_types,
        instruction_data_type_indices,
        instruction_data,
        instruction_data_indices,
        batch_size,
    )

    # Equivalent to embed_instruction_data but avoids nested tensor SDPA.
    instruction_embeddings = []
    for i, instruction_index in enumerate(instruction_indices):
        per_instruction_data = data_tensors[
            (instruction_data_type_indices[:, 0:2] == instruction_index).sum(1) == 2
        ]
        query = torch.cat(
            [instruction_type_embeddings[i].unsqueeze(0), per_instruction_data], dim=0
        ).unsqueeze(0)
        instruction_embeddings.append(
            (query + py_instruction.data_multi_head_attention(query, query, query))
            .sum(1)
            .squeeze(0)
        )

    if instruction_embeddings:
        instruction_embeddings = torch.stack(instruction_embeddings, dim=0)
    else:
        instruction_embeddings = torch.empty(
            (0, py_instruction.dimension_out),
            device=py_instruction.factory_kwargs["device"],
        )

    batched_instructions = []
    for batch_index in range(batch_size):
        per_batch = instruction_embeddings[instruction_indices[:, 0] == batch_index]
        if per_batch.shape[0] == 0:
            batched_instructions.append(
                torch.zeros(
                    py_instruction.dimension_out,
                    device=py_instruction.factory_kwargs["device"],
                    dtype=py_instruction.factory_kwargs["dtype"],
                )
            )
            continue
        positioned = py_instruction._position_embedding(per_batch.unsqueeze(0)).squeeze(
            0
        )
        query = positioned.unsqueeze(0)
        batched_instructions.append(
            (
                query
                + py_instruction.instructions_multi_head_attention(query, query, query)
            )
            .sum(1)
            .squeeze(0)
        )

    return torch.stack(batched_instructions, dim=0)


def test_instruction_embedding_parity(
    py_instruction_data: card_embedding.InstructionDataEmbedding,
    cpp_instruction_data: kumpel_embedding.InstructionDataEmbedding,
    py_shared: card_embedding.SharedEmbeddingHolder,
    cpp_shared: kumpel_embedding.SharedEmbeddingHolder,
    dim: int,
    device: torch.device,
) -> tuple[float, float]:
    instructions_batch = instruction_test_data.instructions_batch

    py_instruction = card_embedding.InstructionEmbedding(
        py_instruction_data, py_shared, dim, device=device
    )
    cpp_instruction = kumpel_embedding.InstructionEmbedding(
        cpp_instruction_data, cpp_shared, dim, device=device
    )
    _with_loaded_weights(py_instruction, cpp_instruction)

    py_out = _python_instruction_forward_reference(py_instruction, instructions_batch)
    cpp_out = cpp_instruction.forward(instructions_batch)
    _assert_close("InstructionEmbedding", py_out, cpp_out)

    py_avg_seconds = _benchmark_forward(
        lambda: _python_instruction_forward_reference(py_instruction, instructions_batch)
    )
    cpp_avg_seconds = _benchmark_forward(lambda: cpp_instruction.forward(instructions_batch))
    return py_avg_seconds, cpp_avg_seconds


def main() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    dim = 32
    device = torch.device("cpu")
    py_shared = card_embedding.SharedEmbeddingHolder(dim, device=device)
    cpp_shared = kumpel_embedding.SharedEmbeddingHolder(dim, device=device)
    _with_loaded_weights(py_shared, cpp_shared)
    py_shared.eval()
    cpp_shared.eval()

    py_instrction_data, cpp_instrction_data = test_instruction_data_embedding_parity(
        py_shared, cpp_shared, dim, device
    )
    py_avg_seconds, cpp_avg_seconds = test_instruction_embedding_parity(
        py_instrction_data, cpp_instrction_data, py_shared, cpp_shared, dim, device
    )
    speedup = py_avg_seconds / cpp_avg_seconds if cpp_avg_seconds > 0 else float("inf")
    print(
        "InstructionEmbedding timing (avg per forward): "
        f"Python={py_avg_seconds * 1e3:.3f} ms, "
        f"C++={cpp_avg_seconds * 1e3:.3f} ms, "
        f"speedup={speedup:.2f}x"
    )
    print("Instruction parity passed.")


if __name__ == "__main__":
    main()
