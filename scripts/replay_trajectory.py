import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', action_type='joint_urx', has_3f_gripper=False, use_camera=False,
            start_joint_positions=tuple([0.019999293696813376, -1.239988205690567, 1.3400126691202097,
                                         4.69998098138019, -1.660014279064388, 0.02001699919159692]))

# env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=False,
#             start_joint_positions=tuple([0.019999293696813376, -1.239988205690567, 1.3400126691202097,
#                                          4.69998098138019, -1.660014279064388, 0.02001699919159692]))

data_interface = DataReplayInterface(env=env)
data_interface.replayTrajectory(traj_file_path='/home/weirdlab/ur_bc/data/traj_21.npz')