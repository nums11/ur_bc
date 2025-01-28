import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.RobomimicDataInterface import RobomimicDataInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

data_interface = RobomimicDataInterface(env_type=UREnv)
data_interface.convertToRobomimicDataset()