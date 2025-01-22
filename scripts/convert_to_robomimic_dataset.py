import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.RobomimicDataInterface import RobomimicDataInterface

data_interface = RobomimicDataInterface()
data_interface.convertToRobomimicDataset()