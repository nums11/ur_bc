import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataCollectionInterface import DataCollectionInterface
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface
from interfaces.GelloTeleopInterface import GelloTeleopInterface
from environments.UREnv import UREnv

env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=False,
            start_joint_positions=tuple([-0.12000154017691589, -1.2999819322497066, 1.4899832468026277,
                                         -1.6300103939608643, -1.610023678771351, -0.16988598712184366]))

# teleop_interface = KeyboardTeleopInterface(env=env)
teleop_interface = GelloTeleopInterface(env=env)

data_interface = DataCollectionInterface(teleop_interface=teleop_interface)
data_interface.startDataCollection(remove_zero_actions=True, collection_freq_hz=10)