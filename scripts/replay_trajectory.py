import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', ee_actions=False, has_3f_gripper=False, use_camera=False,
            start_joint_positions=tuple([0.00987077648325779, -1.4858401502543641, 2.187743336493244,
                                          3.99746649637112, -1.3675774280080697, 4.683919160036989]))

data_interface = DataReplayInterface(env=env)
data_interface.replayTrajectory(traj_file_path='/home/weirdlab/ur_bc/data/traj_24.npz')