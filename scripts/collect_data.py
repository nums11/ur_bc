import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataCollectionInterface import DataCollectionInterface
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', has_3f_gripper=False, use_camera=True,
            start_joint_positions=tuple([0.00987077648325779, -1.4858401502543641, 2.187743336493244,
                                          3.99746649637112, -1.3675774280080697, 4.683919160036989]))

teleop_interface = KeyboardTeleopInterface(env=env)

data_interface = DataCollectionInterface(teleop_interface=teleop_interface)
data_interface.startDataCollection(remove_zero_actions=True, collection_freq_hz=10)