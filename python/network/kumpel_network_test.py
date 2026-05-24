import sys
from pathlib import Path
_REPO = Path("/home/felix/ai/KumpelBrain")  # or Path.cwd().parents[...] if you prefer
_NETWORK = _REPO / "python" / "network"
_PYTESTS = _NETWORK / "pytests"
_CPP_BUILD = _REPO / "cpp" / "build"

for p in (_CPP_BUILD, _PYTESTS, _NETWORK):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
import torch
import kumpel_network
from multi_head_attention import MultiHeadAttentionArgs
import pytests.kumpel_network_json_fixtures as smoke_fixtures

DIM = 12
DIM_INNER = DIM * 4
DIM_INTERACTION_INNER = DIM * 4
DIM_TARGET_INNER = DIM * 4
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 6
game_state_bytes = smoke_fixtures.load_smoke_game_state_bytes()
interaction_bytes = smoke_fixtures.load_smoke_game_interaction_bytes()

device = torch.device("cpu")
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
    attention_args,
    DIM_TARGET_INNER,
    NUM_LAYERS,
    device=device,
    dtype=dtype,
)
model.eval()

with torch.inference_mode():
    scores, _, _, _ = model(game_state_bytes, interaction_bytes)

print(scores)