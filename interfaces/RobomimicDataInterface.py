import os
import h5py
import numpy as np
import json
from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv

class RobomimicDataInterface:
    def __init__(self, env_type):
        self.env_type = env_type
        self.normalize_types = ['min_max', 'mean_std']
        print("Initialized RobomimicDataInterface")
        
    def convertToRobomimicDataset(self, data_dir='/home/weirdlab/ur_bc/data/',
                                  hdf5_path='/home/weirdlab/ur_bc/robomimic_dataset.hdf5',
                                  use_images=False,
                                  normalize=False,
                                  normalize_type='min_max'):
        assert normalize_type in self.normalize_types, "Invalid normalize type valid types are: " + str(self.normalize_types)
        print("RobomimicDataInterface Converting: use_images:", use_images, "normalize:", normalize, "normalize_type:", normalize_type)
        if self.env_type == BimanualUREnv:
            processed_trajectories, num_samples = self._process_trajectories_bimanual(data_dir, use_images, normalize)
            self._create_bimanual_hdf5_dataset(processed_trajectories, hdf5_path, use_images, num_samples)
        elif self.env_type == UREnv:
            processed_trajectories, num_samples = self._process_trajectories(data_dir, use_images, normalize, normalize_type)
            self._create_hdf5_dataset(processed_trajectories, hdf5_path, use_images, num_samples)
    
    def _calculate_min_max(self, data_dir):
        joint_positions = []
        traj_filenames = os.listdir(data_dir)
        for traj_filename in traj_filenames:
            traj_path = os.path.join(data_dir, traj_filename)
            traj = dict(np.load(traj_path, allow_pickle=True).items())
            for t in range(len(traj)):
                obs = traj[str(t)][0]
                joint_positions.append(obs['arm_j'])
        joint_positions = np.array(joint_positions)
        # Min and values for each joint
        min_joint_positions = np.min(joint_positions, axis=0)
        max_joint_positions = np.max(joint_positions, axis=0)
        min_gripper = 0
        max_gripper = 1
        return min_joint_positions, max_joint_positions
    
    def _calculate_mean_std(self, data_dir):
        joint_positions = []
        gripper_values = []
        traj_filenames = os.listdir(data_dir)
        for traj_filename in traj_filenames:
            traj_path = os.path.join(data_dir, traj_filename)
            traj = dict(np.load(traj_path, allow_pickle=True).items())
            for t in range(len(traj)):
                obs = traj[str(t)][0]
                joint_positions.append(obs['arm_j'])
                gripper_values.append(obs['gripper'])
        joint_positions = np.array(joint_positions)
        # Min and values for each joint
        mean_joint_positions = np.mean(joint_positions, axis=0)
        std_joint_positions = np.std(joint_positions, axis=0)
        mean_gripper = np.mean(gripper_values)
        std_gripper = np.std(gripper_values)
        return mean_joint_positions, std_joint_positions, mean_gripper, std_gripper

    def _process_trajectories(self, data_dir, use_images, normalize, normalize_type):
        print("In projcess_trajectories")
        if normalize and normalize_type == 'min_max':
            min_joint_positions, max_joint_positions = self._calculate_min_max(data_dir)
            print("min_joint_positions:", min_joint_positions, "max_joint_positions:", max_joint_positions)
        elif normalize and normalize_type == 'mean_std':
            mean_joint_positions, std_joint_positions, mean_gripper, std_gripper = self._calculate_mean_std(data_dir)
            print("mean_joint_positions:", mean_joint_positions, "std_joint_positions:", std_joint_positions)

        processed_trajectories = []
        num_samples = 0
        traj_filenames = os.listdir(data_dir)
        for traj_filename in traj_filenames:
            traj_path = os.path.join(data_dir, traj_filename)
            traj = dict(np.load(traj_path, allow_pickle=True).items())
            traj_len = len(traj)

            processed_traj = {
                'joint_and_gripper': [],
                'actions': []
            }
            if use_images:
                processed_traj['images'] = []

            for t in range(traj_len):
                if t == traj_len - 1:
                    break
                # Grab current and next observations
                obs = traj[str(t)][0]
                next_obs = traj[str(t+1)][0]

                # Optionally normalize current joint positions
                arm_j = np.array(obs['arm_j'])
                obs_gripper = np.expand_dims(obs['gripper'], axis=0)
                if normalize and normalize_type == 'min_max':
                    arm_j = (arm_j - min_joint_positions) / (max_joint_positions - min_joint_positions)
                elif normalize and normalize_type == 'mean_std':
                    arm_j = (arm_j - mean_joint_positions) / std_joint_positions
                    obs_gripper = (obs_gripper - mean_gripper) / std_gripper

                # Optionally normalize next joint positions
                next_arm_j = np.array(next_obs['arm_j'])
                next_obs_gripper = np.expand_dims(next_obs['gripper'], axis=0)
                if normalize and normalize_type == 'min_max':
                    next_arm_j = (next_arm_j - min_joint_positions) / (max_joint_positions - min_joint_positions)
                elif normalize and normalize_type == 'mean_std':
                    next_arm_j = (next_arm_j - mean_joint_positions) / std_joint_positions
                    next_obs_gripper = (next_obs_gripper - mean_gripper) / std_gripper

                joint_delta = np.subtract(next_arm_j, arm_j)

                joint_and_gripper = np.concatenate((arm_j, obs_gripper))
                concat_action = np.concatenate((joint_delta, next_obs_gripper))
                processed_traj['joint_and_gripper'].append(joint_and_gripper)
                processed_traj['actions'].append(concat_action)

                if use_images:
                    image = obs['image']
                    processed_traj['images'].append(image)

                num_samples += 1
            processed_trajectories.append(processed_traj)
        return processed_trajectories, num_samples

    def _process_trajectories_bimanual(self, data_dir, use_images):
        print("In projcess_trajectories_bimanual")
        processed_trajectories = []
        num_samples = 0
        traj_filenames = os.listdir(data_dir)
        for traj_filename in traj_filenames:
            traj_path = os.path.join(data_dir, traj_filename)
            traj = dict(np.load(traj_path, allow_pickle=True).items())
            traj_len = len(traj)

            processed_traj = {
                'left_joint_and_gripper': [],
                'right_joint_and_gripper': [],
                'actions': []
            }
            if use_images:
                processed_traj['images'] = []

            for t in range(traj_len):
                if t == traj_len - 1:
                    break
                obs = traj[str(t)][0]
                next_obs = traj[str(t+1)][0]
                left_arm_j = np.array(obs['left_arm_j'])
                left_obs_gripper = np.expand_dims(obs['left_gripper'] * 0.02, axis=0)
                right_arm_j = np.array(obs['right_arm_j'])
                right_obs_gripper = np.expand_dims(obs['right_gripper'] * 0.02, axis=0)
                next_left_arm_j = np.array(next_obs['left_arm_j'])
                next_left_obs_gripper = np.expand_dims(next_obs['left_gripper'] * 0.02, axis=0)
                next_right_arm_j = np.array(next_obs['right_arm_j'])
                next_right_obs_gripper = np.expand_dims(next_obs['right_gripper'] * 0.02, axis=0)

                left_joint_delta = np.subtract(next_left_arm_j, left_arm_j)
                right_joint_delta = np.subtract(next_right_arm_j, right_arm_j)

                left_joint_and_gripper = np.concatenate((left_arm_j, left_obs_gripper))
                right_joint_and_gripper = np.concatenate((right_arm_j, right_obs_gripper))
                concat_action = np.concatenate((left_joint_delta, next_left_obs_gripper, right_joint_delta, next_right_obs_gripper))
                processed_traj['left_joint_and_gripper'].append(left_joint_and_gripper)
                processed_traj['right_joint_and_gripper'].append(right_joint_and_gripper)
                processed_traj['actions'].append(concat_action)

                if use_images:
                    image = obs['image']
                    processed_traj['images'].append(image)

                num_samples += 1
            processed_trajectories.append(processed_traj)
        return processed_trajectories, num_samples
    
    def _create_hdf5_dataset(self, processed_trajectories, hdf5_path, use_images, num_samples):
        env_args = {
            "env_name": "MyEnvironment",
            "type": 2,
            "env_kwargs": {},
        }
        env_args_json = json.dumps(env_args)
        with h5py.File(hdf5_path, 'w') as f:
            data_group = f.create_group('data')
            data_group.attrs['total'] = num_samples
            data_group.attrs['env_args'] = env_args_json

            for i, traj in enumerate(processed_trajectories):
                traj_group = data_group.create_group('demo_' + str(i))
                traj_group.attrs['num_samples'] = len(traj['joint_and_gripper'])
                traj_obs_group = traj_group.create_group('obs')
                traj_obs_group.create_dataset('joint_and_gripper', data=traj['joint_and_gripper'], dtype=np.float32)
                if use_images:
                    traj_obs_group.create_dataset('images', data=traj['images'], dtype=np.uint8)
                traj_group.create_dataset('actions', data=traj['actions'], dtype=np.float32)

        print("Created Robomimic dataset at", hdf5_path)

    def _create_bimanual_hdf5_dataset(self, processed_trajectories, hdf5_path, use_images, num_samples):
        env_args = {
            "env_name": "MyEnvironment",
            "type": 2,
            "env_kwargs": {},
        }
        env_args_json = json.dumps(env_args)
        with h5py.File(hdf5_path, 'w') as f:
            data_group = f.create_group('data')
            data_group.attrs['total'] = num_samples
            data_group.attrs['env_args'] = env_args_json

            for i, traj in enumerate(processed_trajectories):
                traj_group = data_group.create_group('demo_' + str(i))
                traj_group.attrs['num_samples'] = len(traj['left_joint_and_gripper'])
                traj_obs_group = traj_group.create_group('obs')
                traj_obs_group.create_dataset('left_joint_and_gripper', data=traj['left_joint_and_gripper'], dtype=np.float32)
                traj_obs_group.create_dataset('right_joint_and_gripper', data=traj['right_joint_and_gripper'], dtype=np.float32)
                if use_images:
                    traj_obs_group.create_dataset('images', data=traj['images'], dtype=np.uint8)
                traj_group.create_dataset('actions', data=traj['actions'], dtype=np.float32)

        print("Created Robomimic dataset at", hdf5_path)