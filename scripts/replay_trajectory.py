import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface

data_interface = DataReplayInterface()
data_interface.replayTrajectory(traj_file_path='/home/weirdlab/ur_bc/data/traj_13.npz')