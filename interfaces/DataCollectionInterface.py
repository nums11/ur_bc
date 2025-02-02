from pynput.keyboard import Listener
from time import sleep
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
import os
import numpy as np

class DataCollectionInterface:
    def __init__(self, teleop_interface):
        self.teleop_interface = teleop_interface
        # Start the pynput keyboard listener
        self.keyboard_listener = Listener(
            on_release=self._on_release
        )

        print("Initialized DataInterface")

    def startDataCollection(self, collection_freq_hz=30, remove_zero_actions=False):
        print("DataInterface collection frequency:", collection_freq_hz, "hz")
        self.keyboard_listener.start()
        self.teleop_interface.start()
        # wait for teleop to start
        while self.teleop_interface.resetting:
            continue

        self.collecting = False
        self.discard = False
        self.save = False
        collection_sleep = 1 / collection_freq_hz
        t = 0
        trajectory = {}
        self._printCollectionMessage()
        while True:
            if self.collecting:
                if self.discard or self.save:
                    if self.discard:
                        print("Discarding Trajectory ---\n")
                    else:
                        print("Saving Trajectory ---\n")
                        self._saveTrajectory(trajectory, remove_zero_actions)
                    t = 0
                    trajectory = {}
                    self.collecting = False
                    self.discard = False
                    self.save = False
                    self.teleop_interface.reset()
                    self._printCollectionMessage()
                    continue

                obs = self.teleop_interface.getObservation()
                trajectory[str(t)] = [obs]
                print("t", t, "obs", obs)
                t += 1
                sleep(collection_sleep)

    def _printCollectionMessage(self):
        print("Press '1' to begin data collection, '2' to discard trajectory, '3' to save trajectory")
        
    def _on_release(self, key):
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
    def _getDatasetFilename(self):
        data_base_dir = '/home/weirdlab/ur_bc/data/'
        num_trajectories = len(os.listdir(data_base_dir))
        filename = data_base_dir + 'traj_' + str(num_trajectories) + '.npz'
        return filename
    
    """ Remove Zero actions from a trajectory"""
    def _removeZeroActions(self, trajectory):
        traj_len = len(trajectory)
        num_zero_actions = 0
        t = 0
        while t < traj_len - 1:
            current_obs = trajectory[str(t)]
            next_t = t + 1
            while next_t < traj_len:
                next_obs = trajectory[str(next_t)]
                if self._isZeroAction(current_obs, next_obs):
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
    
    def _isZeroAction(self, current_obs, next_obs):
        if type(self.teleop_interface.env) == BimanualUREnv:
            return (np.linalg.norm(current_obs[0]['left_arm_pose'] - next_obs[0]['left_arm_pose']) <= 1e-4 and
                    np.linalg.norm(current_obs[0]['right_arm_pose'] - next_obs[0]['right_arm_pose']) <= 1e-4 and
                    current_obs[0]['left_gripper'] == 0 and next_obs[0]['left_gripper'] == 0 and
                    current_obs[0]['right_gripper'] == 0 and next_obs[0]['right_gripper'] == 0)
        elif type(self.teleop_interface.env) == UREnv:
            return (np.linalg.norm(current_obs[0]['arm_pose'] - next_obs[0]['arm_pose']) <= 1e-4 and
                    current_obs[0]['gripper'] == 0 and next_obs[0]['gripper'] == 0)
    
    def _saveTrajectory(self, trajectory, remove_zero_actions):
        filename = self._getDatasetFilename()
        if remove_zero_actions:
            print("DataInterface: Removing Zero actions before saving trajectory")
            trajectory, num_zero_actions = self._removeZeroActions(trajectory)
            print("DataInterface: Removed", num_zero_actions, "actions from trajectory")
        print("DataInterface: Saving trajectory to path", filename)
        np.savez(filename, **trajectory)
        print("\nDataInterface: Finished saving trajectory \n")




            
