"""Parity for FilterEmbedding forward (Python vs C++).

Run: python -m pytest python/network/test_filter_embedding_cpp_parity.py -v
"""

from pathlib import Path
import sys
import os
import tempfile

import torch

import card_embedding
import filter_test_data

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpp" / "build"))
import kumpel_embedding


def _with_loaded_weights(py_model, cpp_model) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        path = fp.name
    try:
        py_model.save_weights(path)
        cpp_model.load_weights(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


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


def _run_case(
    case_name: str,
    py_filter: card_embedding.FilterEmbedding,
    cpp_filter: kumpel_embedding.FilterEmbedding,
    filter_batch: list[dict],
) -> None:
    py_out = py_filter.forward(filter_batch)
    cpp_out = cpp_filter.forward(filter_batch)
    _assert_close(case_name, py_out, cpp_out)


def test_filter_embedding_parity_all_groups() -> None:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cpu")
    dtype = torch.float32
    dim = 32

    py_shared = card_embedding.SharedEmbeddingHolder(dim, device=device, dtype=dtype)
    cpp_shared = kumpel_embedding.SharedEmbeddingHolder(dim, device=device, dtype=dtype)

    py_filter = card_embedding.FilterEmbedding(py_shared, dim, device=device, dtype=dtype)
    cpp_filter = kumpel_embedding.FilterEmbedding(cpp_shared, dim, device=device, dtype=dtype)

    _with_loaded_weights(py_shared, cpp_shared)
    _with_loaded_weights(py_filter, cpp_filter)

    py_shared.eval()
    cpp_shared.eval()
    py_filter.eval()
    cpp_filter.eval()

    for group_name, group_data in filter_test_data.all_test_data.items():
        _run_case(f"group:{group_name}", py_filter, cpp_filter, group_data)

    combined: list[dict] = []
    for group_name in ("simple", "nested", "edge_cases"):
        combined.extend(filter_test_data.all_test_data[group_name])
    _run_case("group:combined_all", py_filter, cpp_filter, combined)
