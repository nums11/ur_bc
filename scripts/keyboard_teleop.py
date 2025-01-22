import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface

teleop = KeyboardTeleopInterface(
    left_arm_start_joint_positions=tuple([0.019659894055907155, -1.5463554047150383, 2.1245277581273756,
                                           4.016029281775736, -1.4222207118732229, 2.317571047171074]),
    right_arm_start_joint_positions=tuple([-0.008423261080850786, -1.414431650807854, -2.2114553494555875,
                                            -0.8639508269995134, -4.436137256425033, 3.3184844991744717]),
    use_camera=False)
teleop.start()

