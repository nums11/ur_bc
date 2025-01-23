import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.ModelEvalInterface import ModelEvalInterface

model_eval = ModelEvalInterface(
    use_camera=False,
    model_path="/home/weirdlab/ur_bc/robomimic/bc_transformer_trained_models/test/20250122160838/models/model_epoch_200.pth",
    # model_path="/home/weirdlab/ur_bc/robomimic/state_bc_models_v1/test/20250122151308/models/model_epoch_25.pth",
    left_arm_start_joint_positions=tuple([0.019659894055907155, -1.5463554047150383, 2.1245277581273756,
                                           4.016029281775736, -1.4222207118732229, 2.317571047171074]),
    right_arm_start_joint_positions=tuple([0.061464341157311926, -1.458567090373423, -2.167427672040368,
                                            -0.9523999794696154, -4.662324173700186, 3.443543061887108])
)
model_eval.evaluate()
