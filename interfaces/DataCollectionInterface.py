from .KeyboardTeleopInterface import KeyboardTeleopInterface
from pynput.keyboard import Listener
from time import sleep
from environments.BimanualUREnv import BimanualUREnv
import os
import numpy as np

class DataCollectionInterface:
    def __init__(self, left_arm_start_joint_positions=None, right_arm_start_joint_positions=None, use_camera=False):
        self.teleop = KeyboardTeleopInterface(
            left_arm_start_joint_positions=left_arm_start_joint_positions,
            right_arm_start_joint_positions=right_arm_start_joint_positions,
            use_camera=use_camera)

        # Start the pynput keyboard listener
        self.keyboard_listener = Listener(
            on_release=self.on_release
        )

        print("Initialized DataInterface")

    def startDataCollection(self, collection_freq_hz=30, remove_zero_actions=False):
        print("DataInterface collection frequency:", collection_freq_hz, "hz")
        self.keyboard_listener.start()
        self.teleop.start()
        # wait for teleop to start
        while self.teleop.resetting:
            continue

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
                    self.teleop.reset()
                    self.printCollectionMessage()
                    continue

                obs = self.teleop.getObservation()
                trajectory[str(t)] = [obs]
                print("t", t, "obs", obs)
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
        t = 0
        while t < traj_len - 1:
            current_obs = trajectory[str(t)]
            next_t = t + 1
            while next_t < traj_len:
                next_obs = trajectory[str(next_t)]
                if (np.linalg.norm(current_obs[0]['left_arm_j'] - next_obs[0]['left_arm_j']) <= 1e-4 and
                    np.linalg.norm(current_obs[0]['right_arm_j'] - next_obs[0]['right_arm_j']) <= 1e-4 and
                    current_obs[0]['left_gripper'] == 0 and next_obs[0]['left_gripper'] == 0 and
                    current_obs[0]['right_gripper'] == 0 and next_obs[0]['right_gripper'] == 0):
                    next_t += 1
                else:
                    break
            if next_t > t + 1:
                for remove_t in range(t + 1, next_t):
                    trajectory.pop(str(remove_t), None)
                    num_zero_actions += 1
            # Adjust the keys of the remaining trajectory
            for shift_t in range(next_t, traj_len):
                trajectory[str(shift_t - (next_t - t - 1))] = trajectory.pop(str(shift_t))
            traj_len -= (next_t - t - 1)
            t += 1
        return trajectory, num_zero_actions
    
    def saveTrajectory(self, trajectory, remove_zero_actions):
        filename = self.getDatasetFilename()
        if remove_zero_actions:
            print("DataInterface: Removing Zero actions before saving trajectory")
            trajectory, num_zero_actions = self.removeZeroActions(trajectory)
            print("DataInterface: Removed", num_zero_actions, "actions from trajectory")
        print("DataInterface: Saving trajectory to path", filename)
        np.savez(filename, **trajectory)
        print("\nDataInterface: Finished saving trajectory \n")




            
