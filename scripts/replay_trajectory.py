import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', action_type='joint_urx', has_3f_gripper=False, use_camera=False,
            start_joint_positions=tuple([0.03002155078287483, -1.249977865684886, 1.3399790174166757,
            -1.6799904601202096, -1.6500190639818983, 2.213486340119196e-05]))

data_interface = DataReplayInterface(env=env)
data_interface.replayTrajectory(traj_file_path='/home/weirdlab/ur_bc/data/traj_12.npz')