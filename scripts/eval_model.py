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

env = UREnv(arm_ip='192.168.1.2', ee_actions=False, has_3f_gripper=False, use_camera=True,
            start_joint_positions=tuple([0.00987077648325779, -1.4858401502543641, 2.187743336493244,
                                          3.99746649637112, -1.3675774280080697, 4.683919160036989]))

model_eval = ModelEvalInterface(
    env=env, blocking=True,
    model_path="/home/weirdlab/ur_bc/robomimic/one_arm_image_bc_models/test/20250128113654/models/model_epoch_100.pth",
)
model_eval.evaluate()
