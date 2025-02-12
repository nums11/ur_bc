import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.ModelEvalInterface import ModelEvalInterface
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

# env = BimanualUREnv(ee_actions=False, use_camera=False,
#     left_arm_start_joint_positions=tuple([0.019659894055907155, -1.5463554047150383, 2.1245277581273756,
#                                            4.016029281775736, -1.4222207118732229, 2.317571047171074]),
#     right_arm_start_joint_positions=tuple([0.061464341157311926, -1.458567090373423, -2.167427672040368,
#                                             -0.9523999794696154, -4.662324173700186, 3.443543061887108]))

env = UREnv(arm_ip='192.168.1.2', action_type='joint_urx', has_3f_gripper=False, use_camera=False,
            start_joint_positions=tuple([0.03002155078287483, -1.249977865684886, 1.3399790174166757,
            -1.6799904601202096, -1.6500190639818983, 2.213486340119196e-05]))

model_eval = ModelEvalInterface(
    env=env, blocking=True,
    model_path="/home/weirdlab/ur_bc/robomimic/gello_pick_v2_state_bc_models/test/20250212135258/models/model_epoch_75.pth",
)
model_eval.evaluate()
