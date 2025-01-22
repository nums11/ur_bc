import os
import h5py
import numpy as np
import json

class RobomimicDataInterface:
    def __init__(self):
        print("Initialized RobomimicDataInterface")
        
    def convertToRobomimicDataset(self, data_dir='/home/weirdlab/ur_bc/data/',
                                  hdf5_path='/home/weirdlab/ur_bc/robomimic_dataset.hdf5',
                                  use_images=False):
        processed_trajectories, num_samples = self._process_trajectories(data_dir, use_images)
        self._create_hdf5_dataset(processed_trajectories, hdf5_path, use_images, num_samples)

    def _process_trajectories(self, data_dir, use_images):
        processed_trajectories = []
        num_samples = 0
        traj_filenames = os.listdir(data_dir)
        for traj_filename in traj_filenames:
            traj_path = os.path.join(data_dir, traj_filename)
            traj = dict(np.load(traj_path, allow_pickle=True).items())
            traj_len = len(traj)

            processed_traj = {
                'joint_ee': [],
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
                next_left_arm_j = np.array(next_obs['left_arm_j'])
                next_left_obs_gripper = np.expand_dims(next_obs['left_gripper'] * 0.02, axis=0)
                if use_images:
                    image = obs['image']
                joint_delta = np.subtract(next_left_arm_j, left_arm_j)

                joint_ee = np.concatenate((left_arm_j, left_obs_gripper))
                concat_action = np.concatenate((joint_delta, next_left_obs_gripper))
                processed_traj['joint_ee'].append(joint_ee)
                processed_traj['actions'].append(concat_action)
                if use_images:
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
                traj_group.attrs['num_samples'] = len(traj['joint_ee'])
                traj_obs_group = traj_group.create_group('obs')
                traj_obs_group.create_dataset('joint_and_gripper', data=traj['joint_ee'], dtype=np.float32)
                if use_images:
                    traj_obs_group.create_dataset('images', data=traj['images'], dtype=np.uint8)
                traj_group.create_dataset('actions', data=traj['actions'], dtype=np.float32)

        print("Created Robomimic dataset at", hdf5_path)