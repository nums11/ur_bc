import numpy as np
import threading
import cv2
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

class DataReplayInterface:
    def __init__(self, env):
        self.env = env
        print("Initialized DataReplayInterface")
    
    def replayTrajectory(self, traj_file_path):
        trajectory = dict(np.load(traj_file_path, allow_pickle=True).items())
        sorted_timesteps = sorted(trajectory.keys(), key=lambda x: int(x))
        print("DataInterface: Replaying Trajectory of length", len(sorted_timesteps))
        for t in sorted_timesteps:
            print("Timestep", t)
            obs = trajectory[str(t)][0]
            action = self._constructActionBasedOnEnv(obs)
            self.env.step(action)

            if 'image' in obs:
                image = obs['image']
                cv2.imwrite("/home/weirdlab/ur_bc/current_obs.jpg", image)

    def _constructActionBasedOnEnv(self, obs):
        action = None
        if type(self.env) == BimanualUREnv:
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
        elif type(self.env) == UREnv:
            arm_j = obs['arm_j']
            gripper = obs['gripper']
            print(arm_j, gripper)
            action = {
                'arm_j': arm_j,
                'gripper': gripper,
            }
        return action

    def _convertGripperToBinary(self, gripper_value):
        return gripper_value > 0.5