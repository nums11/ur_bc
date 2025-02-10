import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

env = BimanualUREnv(left_arm_start_joint_positions=tuple([0.00987077648325779, -1.4858401502543641, 2.187743336493244,
                                          3.99746649637112, -1.3675774280080697, 4.683919160036989]),
                    right_arm_start_joint_positions=tuple([-0.008423261080850786, -1.414431650807854, -2.2114553494555875,
                                            -0.8639508269995134, -4.436137256425033, 3.3184844991744717]),
                    left_arm_has_3f_gripper=False, use_camera=False)


# env = UREnv(arm_ip='192.168.1.2', action_type='ee', has_3f_gripper=False, use_camera=False,
#             start_joint_positions=tuple([0.00987077648325779, -1.4858401502543641, 2.187743336493244,
#                                           3.99746649637112, -1.3675774280080697, 4.683919160036989]))

teleop = KeyboardTeleopInterface(env=env)
teleop.start()

