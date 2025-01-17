import os
import h5py
import numpy as np
import json

"""
Process trajectories converting them into 
obs (j, gripper) -> action (j_delta, gripper)
and removing samples who's distances are too small.
"""
data_dir = '/home/weirdlab/ur_bc/data/'
traj_filenames = os.listdir(data_dir)
features = []
labels = []
processed_trajectories = []
num_samples = 0
for traj_filename in traj_filenames:
    traj_path = data_dir + traj_filename
    traj = dict(np.load(traj_path, allow_pickle=True).items())
    sorted_timesteps = sorted(traj.keys(), key=lambda x: int(x))

    processed_traj = {
        'observations': [],
        'actions': []
    }
    for i, t in enumerate(sorted_timesteps):
        if i == len(sorted_timesteps) - 1:
            continue
        obs, _ = traj[t]
        next_obs, _ = traj[sorted_timesteps[i+1]]
        left_arm_j = np.array(obs['left_arm_j'])
        left_obs_gripper = np.expand_dims(obs['left_gripper'] * 0.02, axis=0)
        next_left_arm_j = np.array(next_obs['left_arm_j'])
        next_left_obs_gripper = np.expand_dims(next_obs['left_gripper'] * 0.02, axis=0)
        joint_delta = np.subtract(next_left_arm_j, left_arm_j)

        # Euclidean distance between joint values
        joint_distance = np.linalg.norm(left_arm_j - next_left_arm_j)
        min_joint_distance = 0
        if joint_distance > min_joint_distance:
            concat_obs = np.concatenate((left_arm_j, left_obs_gripper))
            concat_action = np.concatenate((joint_delta, next_left_obs_gripper))
            features.append(concat_obs)
            labels.append(concat_action)
            processed_traj['observations'].append(concat_obs)
            processed_traj['actions'].append(concat_action)
            num_samples += 1
    processed_trajectories.append(processed_traj)

"""
Create robomimic compatible hdf5 dataset from the processed trajectories
"""
env_args = {
    "env_name": "MyEnvironment",
    "type": 2,
    "env_kwargs": {},
}
env_args_json = json.dumps(env_args)
hdf5_path = '/home/weirdlab/ur_bc/robomimic_compatible_dataset.hdf5'
with h5py.File(hdf5_path, 'w') as f:
    # Create a group named 'data'
    data_group = f.create_group('data')
    data_group.attrs['total'] = num_samples
    data_group.attrs['env_args'] = env_args_json

    # Create group for each trajectory
    for i, traj in enumerate(processed_trajectories):
        traj_group = data_group.create_group('demo_' + str(i))
        traj_group.attrs['num_samples'] = len(traj['observations'])
        # Add observations
        traj_obs_group = traj_group.create_group('obs')
        traj_obs_group.create_dataset('joint_and_gripper', data=traj['observations'], dtype=np.float32)
        # Add actions
        traj_group.create_dataset('actions', data=traj['actions'], dtype=np.float32)

