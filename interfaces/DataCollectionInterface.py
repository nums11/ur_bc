import h5py
from pynput.keyboard import Listener
from time import sleep
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
import os
import numpy as np
import json

class DataCollectionInterface:
    def __init__(self, teleop_interface):
        self.teleop_interface = teleop_interface
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ur_bc_dir = os.path.dirname(script_dir)
        self.data_base_dir = os.path.join(ur_bc_dir, 'data')
        self.metadata_file = os.path.join(self.data_base_dir, 'trajectory_metadata.json')
        
        # Initialize or load metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'total_trajectories': 0,
                'total_samples': 0,
                'max_trajectory_length': 0
            }
            self._save_metadata()

        # Start the pynput keyboard listener
        self.keyboard_listener = Listener(on_release=self._on_release)
        print("Initialized DataInterface")

    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def startDataCollection(self, collection_freq_hz=30, remove_zero_actions=False):
        print("DataInterface collection frequency:", collection_freq_hz, "hz")
        self.keyboard_listener.start()
        
        # Special handling for joint_modbus mode
        if (type(self.teleop_interface.env) == UREnv and 
            self.teleop_interface.env.usesJointModbusActions()):
            # Start in non-blocking mode
            self.teleop_interface.start()
        else:
            # Original blocking behavior for other modes
            self.teleop_interface.start()
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
            sleep(1)

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

    def _getDatasetFilename(self, traj_idx):
        """Get the filename for a trajectory"""
        return os.path.join(self.data_base_dir, f'episode_{traj_idx}.hdf5')
    
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
            if self.teleop_interface.env.usesEEActions():
                return (np.linalg.norm(current_obs[0]['arm_pose'] - next_obs[0]['arm_pose']) <= 1e-4 and
                        current_obs[0]['gripper'] == 0 and next_obs[0]['gripper'] == 0)
            elif self.teleop_interface.env.usesJointModbusActions():
                return (np.linalg.norm(current_obs[0]['arm_j'] - next_obs[0]['arm_j']) <= 1e-4 and
                        current_obs[0]['gripper'] == 0 and next_obs[0]['gripper'] == 0)
    
    def _saveTrajectory(self, trajectory, remove_zero_actions):
        if remove_zero_actions:
            print("DataInterface: Removing Zero actions before saving trajectory")
            trajectory, num_zero_actions = self._removeZeroActions(trajectory)
            print("DataInterface: Removed", num_zero_actions, "actions from trajectory")

        # Get trajectory index from metadata
        traj_idx = self.metadata['total_trajectories']
        traj_len = len(trajectory)
        
        # Update metadata
        self.metadata['total_trajectories'] += 1
        self.metadata['total_samples'] += traj_len
        if traj_len > self.metadata['max_trajectory_length']:
            self.metadata['max_trajectory_length'] = traj_len
            print(f"Updated maximum trajectory length: {traj_len}")
        
        # Create ACT-format HDF5 file
        filename = self._getDatasetFilename(traj_idx)
        
        # Prepare data for ACT format
        data_dict = {
            '/observations/qpos': [],
            '/observations/images/camera': [],
            '/observations/images/wrist_camera': [],
            '/action': [],
        }
        
        # Process each timestep
        for t in range(traj_len - 1):  # -1 since we need next observation for action
            # Current observation
            arm_j = trajectory[str(t)][0]['arm_j']
            gripper = np.expand_dims(trajectory[str(t)][0]['gripper'], axis=0)
            
            # Handle images if available
            if 'image' in trajectory[str(t)][0]:
                image = trajectory[str(t)][0]['image']
                wrist_image = trajectory[str(t)][0]['wrist_image'] if 'wrist_image' in trajectory[str(t)][0] else np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Placeholder if images are not available
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                wrist_image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Next observation (for action)
            next_arm_j = trajectory[str(t + 1)][0]['arm_j']
            next_gripper = np.expand_dims(trajectory[str(t + 1)][0]['gripper'], axis=0)

            # Store current observation and action
            qpos = np.concatenate((arm_j, gripper))
            action = np.concatenate((next_arm_j, next_gripper))
            data_dict['/observations/qpos'].append(qpos)
            data_dict['/observations/images/camera'].append(image)
            data_dict['/observations/images/wrist_camera'].append(wrist_image)
            data_dict['/action'].append(action)
        
        # Add the last state with a null action (replicating the last action)
        # This ensures all observations are included
        t = traj_len - 1
        arm_j = trajectory[str(t)][0]['arm_j']
        gripper = np.expand_dims(trajectory[str(t)][0]['gripper'], axis=0)
        
        if 'image' in trajectory[str(t)][0]:
            image = trajectory[str(t)][0]['image']
            wrist_image = trajectory[str(t)][0]['wrist_image'] if 'wrist_image' in trajectory[str(t)][0] else np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            wrist_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        qpos = np.concatenate((arm_j, gripper))
        # Use the same action as the previous timestep or zeros
        if traj_len > 1:
            action = data_dict['/action'][-1]  # Copy last action
        else:
            action = np.zeros_like(qpos)  # Create zero action
            
        data_dict['/observations/qpos'].append(qpos)
        data_dict['/observations/images/camera'].append(image)
        data_dict['/observations/images/wrist_camera'].append(wrist_image)
        data_dict['/action'].append(action)
        
        # Write to HDF5 file
        with h5py.File(filename, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            # Get actual data dimensions
            num_timesteps = len(data_dict['/observations/qpos'])
            qpos_dim = len(data_dict['/observations/qpos'][0])
            
            # Create datasets
            image.create_dataset('camera', (num_timesteps, 480, 640, 3), dtype='uint8',
                                chunks=(1, 480, 640, 3))
            image.create_dataset('wrist_camera', (num_timesteps, 480, 640, 3), dtype='uint8',
                                chunks=(1, 480, 640, 3))
            obs.create_dataset('qpos', (num_timesteps, qpos_dim))
            root.create_dataset('action', (num_timesteps, qpos_dim))

            # Copy data into datasets
            for name, array in data_dict.items():
                try:
                    # Debug information to identify problematic arrays
                    print(f"Debug: Saving {name} with shape: ", end="")
                    if isinstance(array, list):
                        if len(array) > 0:
                            # Check if all elements have the same shape
                            first_shape = None
                            irregular_indices = []
                            irregular_shapes = []
                            
                            for i, item in enumerate(array):
                                if hasattr(item, 'shape'):
                                    if first_shape is None:
                                        first_shape = item.shape
                                    elif item.shape != first_shape:
                                        irregular_indices.append(i)
                                        irregular_shapes.append(item.shape)
                                        if len(irregular_indices) <= 5:  # Limit to first 5 irregular shapes
                                            print(f"\nWARNING: Item at index {i} has irregular shape {item.shape} (expected {first_shape})")
                            
                            if irregular_indices:
                                print(f"List of length {len(array)} with {len(irregular_indices)} irregular items.")
                                print(f"First few irregular indices: {irregular_indices[:5]}")
                                print(f"Corresponding shapes: {irregular_shapes[:5]}")
                                print(f"Expected shape: {first_shape}")
                            else:
                                print(f"List of length {len(array)} with consistent shape {first_shape}")
                        else:
                            print("Empty list")
                    elif hasattr(array, 'shape'):
                        print(f"Array with shape {array.shape}")
                    else:
                        print(f"Unknown data type: {type(array)}")
                    
                    # Original save operation
                    root[name][...] = array
                except Exception as e:
                    print(f"\n*** ERROR saving {name} ***")
                    print(f"Type: {type(array)}")
                    if isinstance(array, list):
                        print(f"List length: {len(array)}")
                        
                        if len(array) > 0:
                            print(f"First item type: {type(array[0])}")
                            if hasattr(array[0], 'shape'):
                                print(f"First item shape: {array[0].shape}")
                            
                            print("Checking list items for consistency...")
                            unique_shapes = {}
                            for i, item in enumerate(array):
                                if hasattr(item, 'shape'):
                                    shape_key = str(item.shape)
                                    if shape_key not in unique_shapes:
                                        unique_shapes[shape_key] = []
                                    unique_shapes[shape_key].append(i)
                            
                            print(f"Found {len(unique_shapes)} different shapes in the list")
                            for shape, indices in unique_shapes.items():
                                print(f"  - Shape {shape}: {len(indices)} items. Sample indices: {indices[:3]}")
                        
                        print(f"Error details: {str(e)}")
                        
                        # If this is an image array, check for None or invalid entries
                        if name.endswith('/camera') or name.endswith('/wrist_camera'):
                            null_count = 0
                            invalid_count = 0
                            
                            for i, img in enumerate(array):
                                if img is None:
                                    null_count += 1
                                    if null_count <= 3:  # Limit output to first 3 Nones
                                        print(f"  - None found at index {i}")
                                elif not isinstance(img, np.ndarray):
                                    invalid_count += 1
                                    if invalid_count <= 3:  # Limit output to first 3 invalid frames
                                        print(f"  - Non-ndarray at index {i}: {type(img)}")
                                elif len(img.shape) != 3 or img.shape[0] != 480 or img.shape[1] != 640 or img.shape[2] != 3:
                                    invalid_count += 1
                                    if invalid_count <= 3:  # Limit output to first 3 invalid frames
                                        print(f"  - Invalid shape at index {i}: {img.shape}")
                            
                            if null_count > 0:
                                print(f"  - Found {null_count} None values in the list")
                            if invalid_count > 0:
                                print(f"  - Found {invalid_count} invalid frames in the list")
                    
                    raise e
        
        # Save updated metadata
        self._save_metadata()
        
        print(f"\nDataInterface: Saved trajectory {traj_idx} to {filename}")
        print(f"Trajectory length: {traj_len}, Max trajectory length: {self.metadata['max_trajectory_length']}")




            
