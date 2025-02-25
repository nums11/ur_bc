import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataConversionInterface import DataConversionInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

data_interface = DataInterface(env_type=UREnv)
data_interface.convertToDiffusionDataset(use_images=True, normalize=True, normalize_type='min_max')