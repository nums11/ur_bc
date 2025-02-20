import os
import numpy as np
import time
import h5py
from tqdm import tqdm

data_dir='/home/weirdlab/ur_bc/data/'
data_save_dir='/home/weirdlab/ur_bc/act_data/'
data_dict = {
    '/observations/qpos': [],
    '/observations/images/camera': [],
    '/observations/images/wrist_camera': [],
    '/action': [],
}

max_traj_len = 0
traj_filenames = os.listdir(data_dir)
num_trajectories = len(traj_filenames)
all_trajectories = []
# for traj_idx in range(num_trajectories):
#     traj_filename = 'traj_' + str(traj_idx) + '.npz'
#     print((f'Loading trajectory {traj_idx}'))
#     traj_path = os.path.join(data_dir, traj_filename)
#     traj = dict(np.load(traj_path, allow_pickle=True).items())
#     all_trajectories.append(traj)
#     traj_len = len(traj)
#     print(traj_len)
#     if traj_len > max_traj_len:
#         max_traj_len = traj_len

max_traj_len = 900
print(f'Max trajectory length: {max_traj_len}')
# Not sure of the reason for this but it seems to be needed in hdf5 creation
temp_max_traj_len = max_traj_len - 1

for traj_idx in tqdm(range(num_trajectories)):
    traj_filename = 'traj_' + str(traj_idx) + '.npz'
    traj_path = os.path.join(data_dir, traj_filename)
    traj = dict(np.load(traj_path, allow_pickle=True).items())
    traj_len = len(traj)

    for t in range(traj_len):
        if t == traj_len - 1:
            break
        # Grab current and next observations
        obs = traj[str(t)][0]
        next_obs = traj[str(t+1)][0]

        arm_j = np.array(obs['arm_j'])
        obs_gripper = np.expand_dims(obs['gripper'], axis=0)
        image = obs['image']
        wrist_image = obs['wrist_image']

        next_arm_j = np.array(next_obs['arm_j'])
        next_obs_gripper = np.expand_dims(next_obs['gripper'], axis=0)

        # Store current observation and action
        qpos = np.concatenate((arm_j, obs_gripper))
        action = np.concatenate((next_arm_j, next_obs_gripper))
        data_dict['/observations/qpos'].append(qpos)
        data_dict['/observations/images/camera'].append(image)
        data_dict['/observations/images/wrist_camera'].append(wrist_image)
        data_dict['/action'].append(action)

    # Pad the remaining part of the trajectory with the same observation and action
    for _ in range(max_traj_len - traj_len):
        data_dict['/observations/qpos'].append(qpos)
        data_dict['/observations/images/camera'].append(image)
        data_dict['/observations/images/wrist_camera'].append(wrist_image)
        data_dict['/action'].append(action)

    # Covert to hdf5
    t0 = time.time()
    dataset_path = os.path.join(data_save_dir, f'episode_{traj_idx}')
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        image.create_dataset('camera', (temp_max_traj_len, 480, 640, 3), dtype='uint8',
                                        chunks=(1, 480, 640, 3), )
        image.create_dataset('wrist_camera', (temp_max_traj_len, 480, 640, 3), dtype='uint8',
                                chunks=(1, 480, 640, 3), )
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        qpos = obs.create_dataset('qpos', (temp_max_traj_len, 7))
        action = root.create_dataset('action', (temp_max_traj_len, 7))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')

    data_dict = {
        '/observations/qpos': [],
        '/observations/images/camera': [],
        '/observations/images/wrist_camera': [],
        '/action': [],
    }