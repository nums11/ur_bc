import numpy as np
import cv2
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
from time import sleep
import h5py

class DataReplayInterface:
    def __init__(self, env):
        self.env = env
        print("Initialized DataReplayInterface")

    def replayTrajectory(self, hdf5_path, replay_frequency_hz=30):
        self.env.reset()

        if self.env.usesJointModbusActions():
            print("Start ur Program")
            sleep(10)
            print("Running")
            sleep_time = 1 / replay_frequency_hz
        
        with h5py.File(hdf5_path, 'r') as f:
            # Get number of timesteps
            num_samples = f['observations/qpos'].shape[0]
            print(f"DataInterface: Replaying Trajectory, length {num_samples}")
            
            for t in range(num_samples - 1):  # -1 since the last action might be a duplicate
                print(f"Timestep {t}")
                
                # Extract qpos (joint positions and gripper)
                qpos = f['observations/qpos'][t]
                arm_j = qpos[:-1]  # All except the last element (gripper)
                gripper = qpos[-1]  # Last element is gripper
                
                # Construct observation dict for action construction
                obs = {
                    'arm_j': arm_j,
                    'gripper': gripper
                }
                
                # Get the action
                action = self._constructActionBasedOnEnv(obs)
                
                # Execute the action
                self.env.step(action)
                
                # Display images if available
                if 'observations/images/camera' in f:
                    image = f['observations/images/camera'][t]
                    cv2.imshow('Camera View', image)
                    
                    # Display wrist camera if available
                    if 'observations/images/wrist_camera' in f:
                        wrist_image = f['observations/images/wrist_camera'][t]
                        cv2.imshow('Wrist Camera View', wrist_image)
                    
                    # Short wait to display images
                    cv2.waitKey(1)
                
                if self.env.usesJointModbusActions():
                    sleep(sleep_time)
        
        # Clean up CV2 windows
        cv2.destroyAllWindows()
            
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