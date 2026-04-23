"""C++ InstructionEmbedding / ConditionEmbedding forwards on empty batches (card-level empty lists).

Run: python -m pytest python/network/pytests/test_empty_embedding_batches_cpp.py -v
"""

from pathlib import Path
import sys

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"
for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import torch

import kumpel_embedding


def _make_cpp_instruction_and_condition(dim: int, device: torch.device):
    cpp_shared = kumpel_embedding.SharedEmbeddingHolder(dim, device=device)
    cpp_instruction_data = kumpel_embedding.InstructionDataEmbedding(
        cpp_shared, dim, device=device
    )
    cpp_instruction = kumpel_embedding.InstructionEmbedding(
        cpp_instruction_data, cpp_shared, dim, device=device
    )
    cpp_condition = kumpel_embedding.ConditionEmbedding(
        cpp_instruction_data, cpp_shared, dim, device=device
    )
    cpp_shared.eval()
    cpp_instruction_data.eval()
    cpp_instruction.eval()
    cpp_condition.eval()
    return cpp_instruction, cpp_condition


def test_cpp_instruction_embedding_zero_groups_cpu():
    dim = 32
    device = torch.device("cpu")
    cpp_instruction, _ = _make_cpp_instruction_and_condition(dim, device)
    embedded, mask = cpp_instruction.forward([])
    assert embedded.shape == (0, 0, dim)
    assert mask.shape == (0, 0)
    assert embedded.device.type == device.type
    assert mask.dtype == torch.bool


def test_cpp_instruction_embedding_one_empty_group_cpu():
    dim = 32
    device = torch.device("cpu")
    cpp_instruction, _ = _make_cpp_instruction_and_condition(dim, device)
    embedded, mask = cpp_instruction.forward([[]])
    assert embedded.shape == (1, 0, dim)
    assert mask.shape == (1, 0)


def test_cpp_condition_embedding_zero_groups_cpu():
    dim = 32
    device = torch.device("cpu")
    _, cpp_condition = _make_cpp_instruction_and_condition(dim, device)
    embedded, mask = cpp_condition.forward([])
    assert embedded.shape == (0, 0, dim)
    assert mask.shape == (0, 0)


def test_cpp_condition_embedding_one_empty_group_cpu():
    dim = 32
    device = torch.device("cpu")
    _, cpp_condition = _make_cpp_instruction_and_condition(dim, device)
    embedded, mask = cpp_condition.forward([[]])
    assert embedded.shape == (1, 0, dim)
    assert mask.shape == (1, 0)


def test_cpp_card_state_embedding_empty_batch_cpu():
    dim = 32
    device = torch.device("cpu")
    model = kumpel_embedding.CardStateEmbedding(dim, device=device)
    model.eval()
    with torch.inference_mode():
        out = model.forward([])
    assert out.shape == (0, dim)
    assert out.device.type == device.type
