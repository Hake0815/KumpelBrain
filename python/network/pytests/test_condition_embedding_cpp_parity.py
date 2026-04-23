"""Parity for ConditionEmbedding (Python vs C++).

Run: python -m pytest python/network/pytests/test_condition_embedding_cpp_parity.py -v
"""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"
for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import pytest
import torch

import card_embedding
import instruction_test_data
import proto_serialization
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


def _benchmark_forward(
    forward_fn: Callable[[], Any],
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


@contextmanager
def _deterministic_algorithms(enabled: bool):
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )


def run_condition_parity(
    dim: int, device: torch.device, benchmark: bool = False
) -> tuple[float | None, float | None]:
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

    py_out = py_condition.forward(conditions_batch)
    cpp_out = cpp_condition.forward(serialized_conditions_batch)
    _assert_close("ConditionEmbedding", py_out[0], cpp_out[0])
    _assert_close("ConditionEmbedding valid token mask", py_out[1], cpp_out[1])

    if not benchmark:
        return None, None

    py_avg_seconds = _benchmark_forward(
        lambda: py_condition.forward(conditions_batch), device
    )
    cpp_avg_seconds = _benchmark_forward(
        lambda: cpp_condition.forward(serialized_conditions_batch), device
    )
    return py_avg_seconds, cpp_avg_seconds


def test_condition_embedding_parity_cpu() -> None:
    torch.manual_seed(42)
    dim = 32
    run_condition_parity(dim, torch.device("cpu"))


@pytest.mark.parametrize("deterministic", [False, True])
def test_condition_embedding_cuda_benchmark(deterministic: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(42)
    dim = 32
    device = torch.device("cuda")
    with _deterministic_algorithms(deterministic):
        py_avg_seconds, cpp_avg_seconds = run_condition_parity(
            dim, device, benchmark=True
        )
    assert py_avg_seconds is not None
    assert cpp_avg_seconds is not None
