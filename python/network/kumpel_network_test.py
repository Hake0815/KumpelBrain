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
import pytests.kumpel_network_json_fixtures as smoke_fixtures  

DIM = 12
DIM_INNER = DIM * 4 
DIM_INTERACTION_INNER = DIM * 4
NUM_HEADS = 4
NUM_LAYERS = 2
HEAD_DIM = 6
game_state_bytes = smoke_fixtures.load_smoke_game_state_bytes()
interaction_bytes = smoke_fixtures.load_smoke_game_interaction_bytes()


model = kumpel_network.KumpelNetwork(
    DIM,
    DIM_INNER,
    DIM_INTERACTION_INNER,
    HEAD_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    bias=False,
    device=torch.device("cpu"),
    dtype=torch.float32,
)
model.eval()

with torch.inference_mode():
    scores = model(game_state_bytes, interaction_bytes)

print(scores)