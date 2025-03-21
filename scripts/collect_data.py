import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.DataCollectionInterface import DataCollectionInterface
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface
from interfaces.GelloTeleopInterface import GelloTeleopInterface
from environments.UREnv import UREnv

# Peg Insertion
# env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=True,
#             start_joint_positions=tuple([0.02001814887733113, -1.6683640884454487, 1.7006658119817653,
#                                           4.69998098138019, -1.660022823816746, -1.5367661252713862]))
# Twisting
# env = UREnv(arm_ip='192.168.2.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=True,
#             start_joint_positions=tuple([0.2199889837389461, -1.2699844138325087, 1.309975682655657,
#             4.499995139064124, -1.630002999559844, -1.4599217292097606]))


# env = UREnv(arm_ip='192.168.1.2', action_type='ee', has_3f_gripper=False, use_camera=False,
#             start_joint_positions=tuple([-0.012119847983496967, -1.2344485025217573, 1.3694299784791504,
#                                          -1.6486337716066046, -1.5906957964802926, 1.603541160134426]))

# Kitchen
env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=True, use_logitech_camera=True,
            start_joint_positions=tuple([-0.10184682002801448, -1.8316009921757344, 2.2237440184163777,
                -1.9278720721999862, -1.5840280733482741, 0.04111786366790808]))

# teleop_interface = KeyboardTeleopInterface(env=env)
teleop_interface = GelloTeleopInterface(env=env)

data_interface = DataCollectionInterface(teleop_interface=teleop_interface)
data_interface.startDataCollection(remove_zero_actions=False, collection_freq_hz=50)

# while True:
#     pass