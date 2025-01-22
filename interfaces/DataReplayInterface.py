import numpy as np
import threading
import cv2
from environments.BimanualUREnv import BimanualUREnv

class DataReplayInterface:
    def __init__(self, use_camera=False):
        self.env = BimanualUREnv(use_camera=use_camera)
        print("Initialized DataReplayInterface")
    
    def replayTrajectory(self, traj_file_path):
        trajectory = dict(np.load(traj_file_path, allow_pickle=True).items())
        sorted_timesteps = sorted(trajectory.keys(), key=lambda x: int(x))
        print("DataInterface: Replaying Trajectory of length", len(sorted_timesteps))
        for t in sorted_timesteps:
            print("Timestep", t)
            obs = trajectory[str(t)][0]
            print("Observation", obs['left_arm_j'])
            left_j = obs['left_arm_j']
            right_j = obs['right_arm_j']
            left_gripper = obs['left_gripper']
            right_gripper = obs['right_gripper']
            image = None
            if 'image' in obs:
                image = obs['image']
            print(left_j, left_gripper, right_j, right_gripper)
            left_arm_thread = threading.Thread(target=self.armMovementThread,
                                            args=(self.env.left_arm, left_j, left_gripper, image))
            right_arm_thread = threading.Thread(target=self.armMovementThread,
                                            args=(self.env.right_arm, right_j, right_gripper))
            right_arm_thread.start()
            left_arm_thread.start()
            right_arm_thread.join()
            left_arm_thread.join()

    def armMovementThread(self, arm, joints, gripper=None, image=None):
        # action is 6d joint position
        arm.movej(joints)
        if image is not None:
            cv2.imwrite("/home/weirdlab/ur_bc/current_obs.jpg", image)

        if gripper is not None:
            arm.moveRobotiqGripper(gripper)