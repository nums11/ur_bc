from .UREnv import UREnv
from interfaces.URInterface import URInterface
from interfaces.RSCameraInterface import RSCameraInterface
import threading
from time import sleep

class BimanualUREnv:
    def __init__(self, ee_actions=True, right_arm_ip='192.168.2.2', left_arm_ip='192.168.1.2',
                 right_arm_start_joint_positions=None, left_arm_start_joint_positions=None,
                 right_arm_has__3f_gripper=True, left_arm_has_3f_gripper=True, use_camera=False):
        
        if right_arm_start_joint_positions == None:
            right_arm_start_joint_positions = tuple([-0.02262999405073174, -1.1830826636872513, -2.189683323644428,
                                                -1.095669650507004, -4.386985456001609, 3.2958897411425156])
        if left_arm_start_joint_positions == None:
            left_arm_start_joint_positions = tuple([0.04474830963143529, -1.6422924423175793, 1.9634950313912025,
                                                4.267360912521422, -1.4365121397580038, 2.3399834772053114])
        
        self.ee_actions = ee_actions
        self.right_arm_env = UREnv(arm_ip=right_arm_ip, has_3f_gripper=right_arm_has__3f_gripper, use_camera=use_camera,
            start_joint_positions=right_arm_start_joint_positions, robotiq_gripper_port='/dev/ttyUSB0', ee_actions=ee_actions)
        self.left_arm_env = UREnv(arm_ip=left_arm_ip, has_3f_gripper=left_arm_has_3f_gripper, use_camera=False,
            start_joint_positions=left_arm_start_joint_positions, robotiq_gripper_port='/dev/ttyUSB1', ee_actions=ee_actions)

        print("Initialized BimanualUREnv")

    def reset(self):
        print("BimanualUREnv: Resetting")
        self.right_arm_env.reset()
        self.left_arm_env.reset()
        print("BimanualUREnv: Finished Resetting. Start UR Programs")
        return self._getObservation()

    def step(self, action, blocking=True):
        if self.ee_actions:
            self._stepEE(action)
        else:
            self._stepJoints(action, blocking)
        return self._getObservation()
    
    def _stepEE(self, action):
        left_arm_action = {
            'arm_pose': action['left_arm_pose'],
            'gripper': action['left_gripper']
        }
        right_arm_action = {
            'arm_pose': action['right_arm_pose'],
            'gripper': action['right_gripper']
        }
        self.left_arm_env.step(left_arm_action)
        self.right_arm_env.step(right_arm_action)
    
    def _stepJoints(self, action, blocking=True):
        left_arm_action = {
            'arm_j': action['left_arm_j'],
            'gripper': action['left_gripper']
        }
        right_arm_action = {
            'arm_j': action['right_arm_j'],
            'gripper': action['right_gripper']
        }
        self.left_arm_env.step(left_arm_action, blocking)
        self.right_arm_env.step(right_arm_action, blocking)

    def _getObservation(self):
        left_arm_obs = self.left_arm_env._getObservation()
        right_arm_obs = self.right_arm_env._getObservation()
        obs = {
                'left_arm_pose': left_arm_obs['arm_pose'],
                'right_arm_pose': right_arm_obs['arm_pose'],
                'left_arm_j': left_arm_obs['arm_j'],
                'right_arm_j': right_arm_obs['arm_j'],
                'left_gripper': left_arm_obs['gripper'],
                'right_gripper': right_arm_obs['gripper']
                }
        if 'image' in right_arm_obs:
            obs['image'] = right_arm_obs['image']
        return obs