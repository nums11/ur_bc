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

# env = UREnv(arm_ip='192.168.1.2', action_type='joint_urx', has_3f_gripper=False, use_camera=False,
#             start_joint_positions=tuple([0.019999293696813376, -1.239988205690567, 1.3400126691202097,
#                                          4.69998098138019, -1.660014279064388, 0.02001699919159692]))


# env = UREnv(arm_ip='192.168.1.2', action_type='joint_modbus', has_3f_gripper=False, use_camera=False,
#             start_joint_positions=tuple([-0.012119847983496967, -1.2344485025217573, 1.3694299784791504,
#                                          -1.6486337716066046, -1.5906957964802926, 1.603541160134426]))

env = UREnv(arm_ip='192.168.1.2', action_type='joint_urx', has_3f_gripper=False, use_camera=True,
        start_joint_positions=tuple([0.02001814887733113, -1.6683640884454487, 1.7006658119817653,
                                        4.69998098138019, -1.660022823816746, -1.5367661252713862]))

model_eval = ModelEvalInterface(
    env=env,
    # model_path="/home/weirdlab/ur_bc/robomimic/gello_pick_41_demos_std_normalize_with_gripper_state_bc_models/test/20250213220049/models/model_epoch_100.pth",
    model_path="/home/weirdlab/ur_bc/robomimic/diffusion/test_diffusion/20250220182540/models/model_epoch_30.pth",
)
model_eval.evaluate(blocking=True, freq=5, normalize=False, normalize_type='mean_std', diffusion_model=True, frame_stack_size=2)
