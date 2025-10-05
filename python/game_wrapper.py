# Set runtime config files
import pythonnet, os, sys
bin_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'gamecore', 'bin','Debug','net9.0')
if bin_folder not in sys.path:
    sys.path.insert(0, bin_folder)

runtime_config_path = os.path.join(bin_folder,'GameLogicTraining.runtimeconfig.json')
pythonnet.load(runtime="coreclr",runtime_config=runtime_config_path)

import clr
clr.AddReference("GameLogicTraining")

from gamecore.game import IGameController  # replace with your actual namespace/class

class GameControllerWrapper:
    def __init__(self):
        self._game = IGameController.Create("test_log.txt")

