import numpy as np
import threading
import cv2
from environments.BimanualUREnv import BimanualUREnv

class DataReplayInterface:
    def __init__(self, use_camera=False):
        self.env = BimanualUREnv(ee_actions=False, use_camera=use_camera)
        print("Initialized DataReplayInterface")
    
    def replayTrajectory(self, traj_file_path):
        trajectory = dict(np.load(traj_file_path, allow_pickle=True).items())
        sorted_timesteps = sorted(trajectory.keys(), key=lambda x: int(x))
        print("DataInterface: Replaying Trajectory of length", len(sorted_timesteps))
        for t in sorted_timesteps:
            print("Timestep", t)
            obs = trajectory[str(t)][0]
            left_arm_j = obs['left_arm_j']
            right_arm_j = obs['right_arm_j']
            left_gripper = obs['left_gripper']
            right_gripper = obs['right_gripper']
            print(left_arm_j, left_gripper, right_arm_j, right_gripper)

            action = {
                'left_arm_j': left_arm_j,
                'right_arm_j': right_arm_j,
                'left_gripper': self._convertGripperToBinary(left_gripper),
                'right_gripper': self._convertGripperToBinary(right_gripper)
            }
            self.env.step(action)

            if 'image' in obs:
                image = obs['image']
                cv2.imwrite("/home/weirdlab/ur_bc/current_obs.jpg", image)

    def _convertGripperToBinary(self, gripper_value):
        return gripper_value > 0.5