import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataReplayInterface import DataReplayInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=True,
            start_joint_positions=tuple([0.2199889837389461, -1.2699844138325087, 1.309975682655657,
            4.499995139064124, -1.630002999559844, -1.4599217292097606]))

data_interface = DataReplayInterface(env=env)
data_interface.replayTrajectory(hdf5_path='/home/weirdlab/ur_bc/data/raw_demonstrations.h5', traj_idx=10)