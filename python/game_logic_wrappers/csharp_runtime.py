# Set runtime config files
import pythonnet, os, sys

bin_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "gamecore",
    "bin",
    "Debug",
    "net10.0",
)
if bin_folder not in sys.path:
    sys.path.insert(0, bin_folder)

runtime_config_path = os.path.join(bin_folder, "GameLogic.runtimeconfig.json")
pythonnet.load(runtime="coreclr", runtime_config=runtime_config_path)

import clr

clr.AddReference("GameLogic")
