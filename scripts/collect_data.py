import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataCollectionInterface import DataCollectionInterface
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface
from interfaces.GelloTeleopInterface import GelloTeleopInterface
from environments.UREnv import UREnv

# env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=False,
#             start_joint_positions=tuple([0.03002155078287483, -1.249977865684886, 1.3399790174166757,
#             -1.6799904601202096, -1.6500190639818983, 2.213486340119196e-05]))

env = UREnv(arm_ip='192.168.1.2', action_type='ee', has_3f_gripper=False, use_camera=False,
            start_joint_positions=tuple([-0.012119847983496967, -1.2344485025217573, 1.3694299784791504,
                                         -1.6486337716066046, -1.5906957964802926, 1.603541160134426]))

teleop_interface = KeyboardTeleopInterface(env=env)
# teleop_interface = GelloTeleopInterface(env=env)

data_interface = DataCollectionInterface(teleop_interface=teleop_interface)
data_interface.startDataCollection(remove_zero_actions=True, collection_freq_hz=10)
