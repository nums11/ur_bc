import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from robomimic.utils.file_utils import policy_from_checkpoint
from environments.BimanualUREnv import BimanualUREnv
import numpy as np
from time import sleep

class ModelEvalInterface:
    def __init__(self, model_path, use_camera=False, left_arm_start_joint_positions=None, right_arm_start_joint_positions=None):
        self.use_camera = use_camera
        self.env = BimanualUREnv(ee_actions=False, use_camera=self.use_camera,
                                 left_arm_start_joint_positions=left_arm_start_joint_positions,
                                 right_arm_start_joint_positions=right_arm_start_joint_positions)
        self.model, _ = policy_from_checkpoint(ckpt_path=model_path)
        self.model.start_episode()

    def evaluate(self):
        env_obs = self.env.reset()
        while True:
            model_obs = self._convertEnvObsToModelObs(env_obs)
            print("Observed")
            print(model_obs)

            predictions = self.model(model_obs)
            left_arm_delta = predictions[:6]
            left_gripper = self._unnormalizeGripper(predictions[6])
            right_arm_delta = predictions[7:13]
            right_gripper = self._unnormalizeGripper(predictions[13])

            print("Predicted")
            print("left_arm_delta", left_arm_delta)
            print("right_arm_delta", right_arm_delta)
            print("left_gripper", left_gripper)
            print("right_gripper", right_gripper)

            left_arm_j = env_obs['left_arm_j'] + left_arm_delta
            right_arm_j = env_obs['right_arm_j'] + right_arm_delta
            action = {
                'left_arm_j': left_arm_j,
                'right_arm_j': right_arm_j,
                'left_gripper': self._convertGripperToBinary(left_gripper),
                'right_gripper': self._convertGripperToBinary(right_gripper)
            }
            env_obs = self.env.step(action)

    def _convertEnvObsToModelObs(self, obs):
        left_arm_j = obs['left_arm_j']
        right_arm_j = obs['right_arm_j']
        left_obs_gripper = np.expand_dims(obs['left_gripper'], axis=0)
        right_obs_gripper = np.expand_dims(obs['right_gripper'], axis=0)
        model_obs = {
            'left_joint_and_gripper': np.concatenate((left_arm_j, left_obs_gripper)),
            'right_joint_and_gripper': np.concatenate((right_arm_j, right_obs_gripper)),
        }
        if self.use_camera:
            image = obs['image']
            # Change image shape to have channels first
            image = np.transpose(image, (2, 0, 1))
            model_obs['image'] = image
        return model_obs
    
    def _unnormalizeGripper(self, gripper_value):
        return gripper_value / 0.02
    
    def _convertGripperToBinary(self, gripper_value):
        return gripper_value > 0.3