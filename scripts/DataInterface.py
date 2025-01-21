from ContKeyboardTeleopInterface import ContKeyboardTeleopInterface
from pynput.keyboard import Listener
from time import sleep
import os
import numpy as np
import threading
import cv2

class DataInterface:
    def __init__(self, left_arm_start_joint__positions=None, right_arm_start_joint__positions=None):
        self.teleop_interface = ContKeyboardTeleopInterface(
            left_arm_start_joint__positions=left_arm_start_joint__positions,
            right_arm_start_joint__positions=right_arm_start_joint__positions)

        # Start the pynput keyboard listener
        self.keyboard_listener = Listener(
            on_release=self.on_release
        )

        print("Initialized DataInterface")

    def startDataCollection(self, collection_freq_hz=30, remove_zero_actions=False):
        print("DataInterface collection frequency:", collection_freq_hz, "hz")
        self.keyboard_listener.start()
        self.teleop_interface.startTeleop()
        # wait for teleop thread to start
        sleep(2)

        self.collecting = False
        self.discard = False
        self.save = False
        collection_sleep = 1 / collection_freq_hz
        t = 0
        trajectory = {}
        self.printCollectionMessage()
        while True:
            if self.collecting:
                if self.discard or self.save:
                    if self.discard:
                        print("Discarding Trajectory ---\n")
                    else:
                        print("Saving Trajectory ---\n")
                        self.saveTrajectory(trajectory, remove_zero_actions)
                    t = 0
                    trajectory = {}
                    self.collecting = False
                    self.discard = False
                    self.save = False
                    self.teleop_interface.resetArms()
                    self.printCollectionMessage()
                    continue

                obs, action = self.teleop_interface.getLastObsAndAction()
                trajectory[str(t)] = (obs, action)
                print("t", t, "obs", obs, "action", action)
                t += 1
                sleep(collection_sleep)

    def printCollectionMessage(self):
        print("Press '1' to begin data collection, '2' to discard trajectory, '3' to save trajectory")
        
    def on_release(self, key):
        if not hasattr(key, 'char'):
            return
        # Start collecting
        if key.char == '1':
            self.collecting = True
        # Discard Trajectory
        elif key.char == '2':
            if self.collecting:
                self.discard = True
        elif key.char == '3':
            if self.collecting:
                self.save = True

    """ Get the new filename based on the number of existing trajectories """
    def getDatasetFilename(self):
        data_base_dir = '/home/weirdlab/ur_bc/data/'
        num_trajectories = len(os.listdir(data_base_dir))
        filename = data_base_dir + 'traj_' + str(num_trajectories) + '.npz'
        return filename
    
    """ Remove Zero actions from a trajectory"""
    def removeZeroActions(self, trajectory):
        traj_len = len(trajectory)
        num_zero_actions = 0
        for t in range(traj_len):
            _, action = trajectory[str(t)]
            if self.isZeroAction(action):
                print("DataInterface: Found zero action at timestep", t)
                num_zero_actions += 1
                trajectory.pop(str(t), None)
        return trajectory, num_zero_actions
    
    def isZeroAction(self, action):
        # If any of the deltas are non zero this is not a zero action
        if np.any(action['left_arm_delta']) or np.any(action['right_arm_delta']):
            return False
        # If any of the grippers are closed this is not a zero action
        if action['left_gripper'] or action['right_gripper']:
            return False
        return True 

    def saveTrajectory(self, trajectory, remove_zero_actions):
        filename = self.getDatasetFilename()
        if remove_zero_actions:
            print("DataInterface: Removing Zero actions before saving trajectory")
            trajectory, num_zero_actions = self.removeZeroActions(trajectory)
            print("DataInterface: Removed", num_zero_actions, "actions from trajectory")
        print("DataInterface: Saving trajectory to path", filename)
        np.savez(filename, **trajectory)
        print("\nDataInterface: Finished saving trajectory \n")

    def replayTrajectory(self, traj_file_path, joint_position_replay=False):
        trajectory = dict(np.load(traj_file_path, allow_pickle=True).items())
        sorted_timesteps = sorted(trajectory.keys(), key=lambda x: int(x))
        print("DataInterface: Replaying Trajectory of length", len(sorted_timesteps))
        for t in sorted_timesteps:
            print("Timestep", t)
            obs, action = trajectory[str(t)]
            if joint_position_replay:
                left_action = obs['left_arm_j']
                right_action = obs['right_arm_j']
                left_gripper = obs['left_gripper']
                right_gripper = obs['right_gripper']
                image = obs['image']
            else:
                left_action = action['left_arm_delta']
                right_action = action['right_arm_delta']
                left_gripper = action['left_gripper']
                right_gripper = action['right_gripper']
                print(left_action, left_gripper, right_action, right_gripper)
            left_arm_thread = threading.Thread(target=self.armMovementThread,
                                            args=(self.teleop_interface.left_arm, left_action, image, joint_position_replay, left_gripper))
            # right_arm_thread = threading.Thread(target=self.armMovementThread,
            #                                 args=(self.teleop_interface.right_arm, right_action, joint_position_replay, right_gripper))
            # right_arm_thread.start()
            left_arm_thread.start()
            # right_arm_thread.join()
            left_arm_thread.join()

    def armMovementThread(self, arm, action, image=None, joint_position_replay=False, gripper=None):
        if joint_position_replay:
            # action is 6d joint position
            arm.movej(action)
            if image is not None:
                cv2.imwrite("/home/weirdlab/ur_bc/current_obs.jpg", image)

        else:
            # action is 6d ee position
            # action = np.zeros(6)
            # print("overwriting action", action)
            pose = arm.getPose()
            pose += action
            arm.movejInvKin(pose)
        if gripper is not None:
            arm.moveRobotiqGripper(gripper)


            
