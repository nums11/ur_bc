import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', action_type='joint_urx', has_3f_gripper=False, use_camera=True,
            start_joint_positions=tuple([0.02001814887733113, -1.6683640884454487, 1.7006658119817653,
                                         4.69998098138019, -1.660022823816746, -1.5367661252713862]))

data_interface = DataReplayInterface(env=env)
data_interface.replayTrajectory(traj_file_path='/home/weirdlab/ur_bc/data/traj_3.npz')