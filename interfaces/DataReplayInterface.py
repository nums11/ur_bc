import numpy as np
import threading
import cv2
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
from pynput.keyboard import Listener
from time import sleep
import h5py

class DataReplayInterface:
    def __init__(self, env):
        self.env = env
        print("Initialized DataReplayInterface")

    def replayTrajectory(self, hdf5_path, traj_idx, replay_frequency_hz=10):
        self.env.reset()

        if self.env.usesJointModbusActions():
            print("Start ur Program")
            sleep(10)
            print("Running")
            sleep_time = 1 / replay_frequency_hz
        
        with h5py.File(hdf5_path, 'r') as f:
            traj_group = f['data'][f'traj_{traj_idx}']
            num_samples = traj_group.attrs['num_samples']
            print("DataInterface: Replaying Trajectory of length", num_samples)

            for t in range(num_samples):
                print("Timestep", t)
                obs = {key: traj_group[key][t] for key in traj_group.keys()}
                action = self._constructActionBasedOnEnv(obs)
                self.env.step(action)

                if 'image' in obs:
                    image = obs['image']
                    wrist_image = obs['wrist_image']
                    cv2.imwrite("/home/weirdlab/ur_bc/image_obs.jpg", image)
                    cv2.imwrite("/home/weirdlab/ur_bc/wrist_image_obs.jpg", wrist_image)

                if self.env.usesJointModbusActions():
                    sleep(sleep_time)
            
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
        return gripper_value >= 0.5