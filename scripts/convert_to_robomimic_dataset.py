import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataInterface import DataInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

data_interface = DataInterface(env_type=UREnv)
data_interface.convertToRobomimicDataset(use_images=False, normalize=True, normalize_type='mean_std')