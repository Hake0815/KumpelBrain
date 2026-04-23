"""Parity for nesting filter traversal / flatten / reduce (Python vs C++).

Run: python -m pytest python/network/test_nesting_cpp_filter_parity.py -v
"""

from pathlib import Path
import sys

import torch

import filter_test_data
import nesting

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cpp" / "build"))
import kumpel_embedding


def _assert_equal(name: str, a, b) -> None:
    if a != b:
        raise AssertionError(f"{name} mismatch:\npython={a}\ncpp={b}")


def _assert_reduce_close(name: str, py_values, cpp_values, atol: float = 1e-6) -> None:
    if len(py_values) != len(cpp_values):
        raise AssertionError(
            f"{name} length mismatch:\npython={len(py_values)}\ncpp={len(cpp_values)}"
        )
    for idx, (py_value, cpp_value) in enumerate(zip(py_values, cpp_values)):
        if not torch.allclose(py_value, cpp_value, atol=atol):
            raise AssertionError(
                f"{name}[{idx}] mismatch:\npython={py_value}\ncpp={cpp_value}"
            )


def _combine(values, operator):
    stacked = torch.stack(values, dim=0)
    combined = torch.sum(stacked, dim=0)
    if operator is not None:
        combined = combined + float(operator)
    return combined


def _run_case(case_name: str, case_data: list[dict]) -> None:
    py_traverse = list(nesting.traverse_filter(case_data))
    cpp_traverse = kumpel_embedding.nesting_traverse_filter(case_data)
    _assert_equal(f"{case_name} traverse_filter", py_traverse, cpp_traverse)

    py_flattened, py_groups, py_operators = nesting.flatten(
        case_data, nesting.traverse_filter
    )
    cpp_flattened, cpp_groups, cpp_operators = kumpel_embedding.nesting_flatten_filter(
        case_data
    )
    _assert_equal(f"{case_name} flattened_input", py_flattened, cpp_flattened)
    _assert_equal(f"{case_name} groups", py_groups, cpp_groups)
    _assert_equal(f"{case_name} operators", py_operators, cpp_operators)

    py_flattened_tensor = torch.tensor(py_flattened, dtype=torch.float32)
    py_reduced = nesting.reduce(py_flattened_tensor, py_groups, py_operators, _combine)
    cpp_reduced = kumpel_embedding.nesting_reduce(
        py_flattened_tensor, cpp_groups, cpp_operators, _combine
    )
    _assert_reduce_close(f"{case_name} reduce", py_reduced, cpp_reduced)


def test_nesting_filter_parity_all_groups() -> None:
    for case_name, case_data in filter_test_data.all_test_data.items():
        _run_case(case_name, case_data)
