#!/usr/bin/env python3
"""Regenerate golden tensors for CardEmbedding.forward (CPU and CUDA).

Run from repo root or from python/network (uses repo-relative paths):

  .venv/bin/python python/network/generate_card_embedding_forward_goldens.py

Requires built kumpel_embedding under cpp/build (same Python ABI as this interpreter).

CUDA file is written only when torch.cuda.is_available().
If CUDA raises nondeterminism errors, try:
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NETWORK_DIR = Path(__file__).resolve().parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"
_FIXTURES_DIR = _NETWORK_DIR / "fixtures"

sys.path.insert(0, str(_NETWORK_DIR))
sys.path.insert(0, str(_CPP_BUILD))

import card_embedding_forward_fixtures as fixtures  # noqa: E402
import kumpel_embedding  # noqa: E402

GOLDEN_SEED = 42


@contextlib.contextmanager
def _deterministic_algorithms(enabled: bool):
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def _seed_for_device(device: torch.device) -> None:
    torch.manual_seed(GOLDEN_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(GOLDEN_SEED)


def _build_model(device: torch.device) -> kumpel_embedding.CardEmbedding:
    _seed_for_device(device)
    m = kumpel_embedding.CardEmbedding(
        fixtures.FIXTURE_DIMENSION_OUT, device=device, dtype=torch.float32
    )
    m.eval()
    return m


def _forward_case(model, card_bytes: list[bytes]) -> torch.Tensor:
    with torch.inference_mode():
        embedding, _adjacency = model.forward(card_bytes)
        return embedding


def generate_for_device(device: torch.device) -> dict[str, torch.Tensor]:
    _seed_for_device(device)
    model = _build_model(device)
    gold: dict[str, torch.Tensor] = {}
    for case_id in sorted(fixtures.FIXTURE_CASES.keys()):
        cards = fixtures.FIXTURE_CASES[case_id]
        out = _forward_case(model, cards)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gold[case_id] = out.detach().cpu().contiguous()
    return gold


def main() -> None:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    with _deterministic_algorithms(True):
        cpu_path = _FIXTURES_DIR / "card_embedding_forward_golden_cpu.pt"
        cpu_gold = generate_for_device(torch.device("cpu"))
        torch.save(cpu_gold, cpu_path)
        print(f"Wrote {cpu_path} ({len(cpu_gold)} cases)")

        if torch.cuda.is_available():
            cuda_path = _FIXTURES_DIR / "card_embedding_forward_golden_cuda.pt"
            cuda_gold = generate_for_device(torch.device("cuda", 0))
            torch.save(cuda_gold, cuda_path)
            print(f"Wrote {cuda_path} ({len(cuda_gold)} cases)")
        else:
            print("CUDA not available; skipped card_embedding_forward_golden_cuda.pt")


if __name__ == "__main__":
    main()
