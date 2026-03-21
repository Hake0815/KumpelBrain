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


def _set_constant_parameters(model, value: float = 0.01) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def _benchmark_forward(
    forward_fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup_runs: int = 20,
    runs: int = 200,
) -> float:
    def _synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup_runs):
        forward_fn()

    _synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        forward_fn()
    _synchronize()
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
    serialized_filter_data = [
        proto_serialization.serialize_filter_payload(filter_payload)
        for filter_payload in instruction_data[4]
    ]
    cpp_instruction_data_input = list(instruction_data)
    cpp_instruction_data_input[4] = serialized_filter_data
    cpp_out = cpp_instruction_data.forward(
        instruction_indices,
        instruction_data_types,
        instruction_data_type_indices,
        tuple(cpp_instruction_data_input),
        instruction_data_indices,
        batch_size,
    )
    _assert_close("InstructionDataEmbedding", py_out, cpp_out)

    return py_instruction_data, cpp_instruction_data


def test_instruction_embedding_parity(
    py_instruction_data: card_embedding.InstructionDataEmbedding,
    cpp_instruction_data: kumpel_embedding.InstructionDataEmbedding,
    py_shared: card_embedding.SharedEmbeddingHolder,
    cpp_shared: kumpel_embedding.SharedEmbeddingHolder,
    dim: int,
    device: torch.device,
    benchmark: bool = False,
) -> tuple[float | None, float | None]:
    instructions_batch = instruction_test_data.instructions_batch
    serialized_instructions_batch = proto_serialization.serialize_instruction_batches(
        instructions_batch
    )

    py_instruction = card_embedding.InstructionEmbedding(
        py_instruction_data, py_shared, dim, device=device
    )
    cpp_instruction = kumpel_embedding.InstructionEmbedding(
        cpp_instruction_data, cpp_shared, dim, device=device
    )
    _with_loaded_weights(py_instruction, cpp_instruction)

    py_out = py_instruction.forward(instructions_batch)
    cpp_out = cpp_instruction.forward(serialized_instructions_batch)
    _assert_close("InstructionEmbedding", py_out, cpp_out)

    if not benchmark:
        return None, None

    py_avg_seconds = _benchmark_forward(lambda: py_instruction.forward(instructions_batch), device)
    cpp_avg_seconds = _benchmark_forward(
        lambda: cpp_instruction.forward(serialized_instructions_batch), device
    )
    return py_avg_seconds, cpp_avg_seconds


def run_instruction_parity(
    dim: int, device: torch.device, benchmark: bool = False
) -> tuple[float | None, float | None]:
    py_shared = card_embedding.SharedEmbeddingHolder(dim, device=device)
    cpp_shared = kumpel_embedding.SharedEmbeddingHolder(dim, device=device)
    _with_loaded_weights(py_shared, cpp_shared)
    py_shared.eval()
    cpp_shared.eval()

    py_instrction_data, cpp_instrction_data = test_instruction_data_embedding_parity(
        py_shared, cpp_shared, dim, device
    )
    return test_instruction_embedding_parity(
        py_instrction_data,
        cpp_instrction_data,
        py_shared,
        cpp_shared,
        dim,
        device,
        benchmark=benchmark,
    )


def main() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    dim = 32
    run_instruction_parity(dim, torch.device("cpu"))
    print("Instruction CPU fallback parity passed.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        py_avg_seconds, cpp_avg_seconds = run_instruction_parity(
            dim, device, benchmark=True
        )
        assert py_avg_seconds is not None
        assert cpp_avg_seconds is not None
        speedup = (
            py_avg_seconds / cpp_avg_seconds if cpp_avg_seconds > 0 else float("inf")
        )
        print(
            "InstructionEmbedding timing (avg per forward): "
            f"Python={py_avg_seconds * 1e3:.3f} ms, "
            f"C++={cpp_avg_seconds * 1e3:.3f} ms, "
            f"speedup={speedup:.2f}x"
        )
    else:
        print("Instruction CUDA timing skipped: CUDA is not available.")

    print("Instruction parity passed.")


if __name__ == "__main__":
    main()
