"""Construct shared KumpelNetwork + Selector for self-play inference."""

from __future__ import annotations

import os

import torch

from network.kumpel_network import KumpelNetwork
from network.multi_head_attention import MultiHeadAttentionArgs
from network.selector import Selector

DIMENSION_OUT = 128
NUM_LAYERS = 12


def resolve_compute_device(explicit: torch.device | str | None = None) -> torch.device:
    """Self-play device: KUMPEL_DEVICE env, explicit arg, else cpu (batch-1 inference)."""
    if explicit is not None:
        return torch.device(explicit)
    env = os.environ.get("KUMPEL_DEVICE", "").strip().lower()
    if env in ("cuda", "gpu"):
        return torch.device("cuda")
    if env in ("cpu", ""):
        return torch.device("cpu")
    return torch.device(env)


def create_self_play_models(
    compute_device: torch.device | str | None = None,
    *,
    embed_on_compute_device: bool = False,
) -> tuple[KumpelNetwork, Selector, torch.device]:
    """Build eval-ready network and selector; caller should reuse across games."""
    device = resolve_compute_device(compute_device)
    embedding_device = device if embed_on_compute_device else torch.device("cpu")
    dimension_state_inner = DIMENSION_OUT * 4
    dimension_interaction_inner = DIMENSION_OUT * 4
    dimension_target_inner = DIMENSION_OUT * 4

    attention_args = MultiHeadAttentionArgs(
        DIMENSION_OUT,
        DIMENSION_OUT,
        DIMENSION_OUT,
        32,
        4,
        bias=False,
        device=device,
    )
    network = KumpelNetwork(
        DIMENSION_OUT,
        dimension_state_inner,
        dimension_interaction_inner,
        attention_args,
        attention_args,
        NUM_LAYERS,
        device=device,
        embedding_device=embedding_device,
    )
    network.eval()

    # Selector on same device as compute for self-play (avoids cross-device copies).
    selector = Selector(
        DIMENSION_OUT,
        dimension_target_inner,
        attention_args,
        device=device,
    )
    selector.eval()

    return network, selector, device
