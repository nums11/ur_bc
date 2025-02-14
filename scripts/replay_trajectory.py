import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', action_type='joint_urx', has_3f_gripper=False, use_camera=False,
            start_joint_positions=tuple([-0.012119847983496967, -1.2344485025217573, 1.3694299784791504,
                                         -1.6486337716066046, -1.5906957964802926, 1.603541160134426]))

data_interface = DataReplayInterface(env=env)
data_interface.replayTrajectory(traj_file_path='/home/weirdlab/ur_bc/data/traj_10.npz')