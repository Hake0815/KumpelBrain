"""Manual smoke script for KumpelNetwork (not part of pytest).

Run from repo root:
  python python/network/kumpel_network_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_NETWORK = _REPO / "python" / "network"
_PYTESTS = _NETWORK / "pytests"
_CPP_BUILD = _REPO / "cpp" / "build"

for p in (_CPP_BUILD, _PYTESTS, _NETWORK):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import torch  # noqa: E402

import kumpel_network  # noqa: E402
import kumpel_network_json_fixtures as smoke_fixtures  # noqa: E402
from multi_head_attention import MultiHeadAttentionArgs  # noqa: E402

DIM = 12
DIM_INNER = DIM * 4
DIM_INTERACTION_INNER = DIM * 4
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 6


def main() -> None:
    game_state_bytes = smoke_fixtures.load_smoke_game_state_bytes()
    interaction_bytes = smoke_fixtures.load_smoke_game_interaction_bytes()

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    dtype = torch.float32
    attention_args = MultiHeadAttentionArgs(
        DIM, DIM, DIM, HEAD_DIM, NUM_HEADS, bias=False, device=device, dtype=dtype
    )

    model = kumpel_network.KumpelNetwork(
        DIM,
        DIM_INNER,
        DIM_INTERACTION_INNER,
        attention_args,
        attention_args,
        NUM_LAYERS,
        device=device,
        dtype=dtype,
    )
    model.eval()

    with torch.inference_mode():
        scores, _, _, _ = model(game_state_bytes, interaction_bytes)

    print(scores)
    print(torch.argmax(scores, dim=0).item())


if __name__ == "__main__":
    main()
