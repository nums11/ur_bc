import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.GelloTeleopInterface import GelloTeleopInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

# env = BimanualUREnv(left_arm_start_joint_positions=tuple([0.00987077648325779, -1.4858401502543641, 2.187743336493244,
#                                           3.99746649637112, -1.3675774280080697, 4.683919160036989]),
#                     right_arm_start_joint_positions=tuple([-0.008423261080850786, -1.414431650807854, -2.2114553494555875,
#                                             -0.8639508269995134, -4.436137256425033, 3.3184844991744717]),
#                     left_arm_has_3f_gripper=False, use_camera=False)

# Vention table joint positions
# env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=False,
#             start_joint_positions=tuple([2.22746793574166e-05, -1.5708081003042818, 1.5707789608553613,
#                                          -1.5708197319078439, -1.5707769600749, -1.792811706113895e-05]))

# Kitchen joint positions
env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus',
            has_3f_gripper=False, use_camera=False, use_current_joint_positions=True)

teleop = GelloTeleopInterface(env=env)
teleop.start()

