import os
import numpy as np
import time
import h5py
from tqdm import tqdm
import argparse

def convert_trajectories(start_idx=0):
    input_hdf5_path = '/home/weirdlab/ur_bc/data/raw_demonstrations.h5'
    data_save_dir = '/media/weirdlab/5f6018bf-feb1-4eb4-bdd0-acb9dc9e0422/act_data/'

    data_dict = {
        '/observations/qpos': [],
        '/observations/images/camera': [],
        '/observations/images/wrist_camera': [],
        '/action': [],
    }

    # Read trajectories from HDF5 file
    with h5py.File(input_hdf5_path, 'r') as f:
        data_group = f['data']
        num_trajectories = data_group.attrs['total_trajectories']
        max_traj_len = max(data_group[f'traj_{i}'].attrs['num_samples'] 
                          for i in range(num_trajectories))
        
        print(f'Processing trajectories from index {start_idx} to {num_trajectories-1}')
        print(f'Max trajectory length: {max_traj_len}')
        temp_max_traj_len = max_traj_len - 1  # Required for HDF5 creation
        
        for traj_idx in tqdm(range(start_idx, num_trajectories)):
            traj_group = data_group[f'traj_{traj_idx}']
            traj_len = traj_group.attrs['num_samples']

            # Process each timestep
            for t in range(traj_len - 1):  # -1 since we need next observation for action
                # Current observation
                arm_j = traj_group['arm_j'][t]
                gripper = np.expand_dims(traj_group['gripper'][t], axis=0)
                image = traj_group['image'][t]
                wrist_image = traj_group['wrist_image'][t]

                # Next observation (for action)
                next_arm_j = traj_group['arm_j'][t + 1]
                next_gripper = np.expand_dims(traj_group['gripper'][t + 1], axis=0)

                # Store current observation and action
                qpos = np.concatenate((arm_j, gripper))
                action = np.concatenate((next_arm_j, next_gripper))
                data_dict['/observations/qpos'].append(qpos)
                data_dict['/observations/images/camera'].append(image)
                data_dict['/observations/images/wrist_camera'].append(wrist_image)
                data_dict['/action'].append(action)

            # Pad trajectory to fixed length
            for _ in range(max_traj_len - traj_len):
                data_dict['/observations/qpos'].append(qpos)
                data_dict['/observations/images/camera'].append(image)
                data_dict['/observations/images/wrist_camera'].append(wrist_image)
                data_dict['/action'].append(action)

            # Save to ACT format
            t0 = time.time()
            dataset_path = os.path.join(data_save_dir, f'episode_{traj_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                image = obs.create_group('images')
                image.create_dataset('camera', (temp_max_traj_len, 480, 640, 3), dtype='uint8',
                                    chunks=(1, 480, 640, 3))
                image.create_dataset('wrist_camera', (temp_max_traj_len, 480, 640, 3), dtype='uint8',
                                    chunks=(1, 480, 640, 3))
                qpos = obs.create_dataset('qpos', (temp_max_traj_len, 7))
                action = root.create_dataset('action', (temp_max_traj_len, 7))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')

            # Reset data dict for next trajectory
            data_dict = {
                '/observations/qpos': [],
                '/observations/images/camera': [],
                '/observations/images/wrist_camera': [],
                '/action': [],
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0, 
                        help='Starting trajectory index')
    args = parser.parse_args()
    
    convert_trajectories(args.start_idx)