#!/usr/bin/env python3
"""Regenerate golden tensors for GameEmbedding embedGameState / embedGameInteraction (CPU and CUDA).

Run from repo root:

  .venv/bin/python python/network/pytests/generate_game_embedding_goldens.py

Requires built kumpel_embedding under cpp/build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PYTESTS_DIR = Path(__file__).resolve().parent
_NETWORK_SRC_DIR = _PYTESTS_DIR.parent
_REPO_ROOT = _NETWORK_SRC_DIR.parent.parent
_CPP_BUILD = _REPO_ROOT / "cpp" / "build"

for _p in (_CPP_BUILD, _PYTESTS_DIR, _NETWORK_SRC_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import game_embedding_fixtures as fixtures  # noqa: E402
import kumpel_embedding  # noqa: E402
from golden_test_utils import FIXTURES_DIR, deterministic_algorithms, seed_for_device  # noqa: E402


def _build_model(device: torch.device) -> kumpel_embedding.GameEmbedding:
    seed_for_device(device)
    model = kumpel_embedding.GameEmbedding(
        fixtures.FIXTURE_DIMENSION_OUT, device=device, dtype=torch.float32
    )
    model.eval()
    return model


def generate_game_state_for_device(device: torch.device) -> dict[str, torch.Tensor]:
    seed_for_device(device)
    model = _build_model(device)
    gold: dict[str, torch.Tensor] = {}
    for case_id in sorted(fixtures.EMBED_GAME_STATE_CASES.keys()):
        payload = fixtures.EMBED_GAME_STATE_CASES[case_id]
        with torch.inference_mode():
            embedding, _card_indices = model.embedGameState(payload)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gold[case_id] = embedding.detach().cpu().contiguous()
    return gold


def generate_game_interaction_for_device(device: torch.device) -> dict[str, torch.Tensor]:
    seed_for_device(device)
    model = _build_model(device)
    gold: dict[str, torch.Tensor] = {}
    for case_id in sorted(fixtures.EMBED_GAME_INTERACTION_CASES.keys()):
        game_state_bytes, interaction_bytes = fixtures.EMBED_GAME_INTERACTION_CASES[case_id]
        with torch.inference_mode():
            game_state_embedding, indices = model.embedGameState(game_state_bytes)
            cards = fixtures.extract_card_embeddings(game_state_embedding)
            out = model.embedGameInteraction(interaction_bytes, indices, cards)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gold[case_id] = out.detach().cpu().contiguous()
    return gold


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    with deterministic_algorithms(True):
        state_cpu_path = FIXTURES_DIR / "game_embedding_embed_game_state_golden_cpu.pt"
        state_cpu_gold = generate_game_state_for_device(torch.device("cpu"))
        torch.save(state_cpu_gold, state_cpu_path)
        print(f"Wrote {state_cpu_path} ({len(state_cpu_gold)} cases)")

        interaction_cpu_path = FIXTURES_DIR / "game_embedding_embed_game_interaction_golden_cpu.pt"
        interaction_cpu_gold = generate_game_interaction_for_device(torch.device("cpu"))
        torch.save(interaction_cpu_gold, interaction_cpu_path)
        print(f"Wrote {interaction_cpu_path} ({len(interaction_cpu_gold)} cases)")

        if torch.cuda.is_available():
            state_cuda_path = FIXTURES_DIR / "game_embedding_embed_game_state_golden_cuda.pt"
            state_cuda_gold = generate_game_state_for_device(torch.device("cuda", 0))
            torch.save(state_cuda_gold, state_cuda_path)
            print(f"Wrote {state_cuda_path} ({len(state_cuda_gold)} cases)")

            interaction_cuda_path = FIXTURES_DIR / "game_embedding_embed_game_interaction_golden_cuda.pt"
            interaction_cuda_gold = generate_game_interaction_for_device(
                torch.device("cuda", 0)
            )
            torch.save(interaction_cuda_gold, interaction_cuda_path)
            print(f"Wrote {interaction_cuda_path} ({len(interaction_cuda_gold)} cases)")
        else:
            print("CUDA not available; skipped game_embedding *_golden_cuda.pt")


if __name__ == "__main__":
    main()
