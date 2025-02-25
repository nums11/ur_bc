import h5py
import cv2
import argparse
import numpy as np

def view_trajectory_images(hdf5_path, traj_idx, timestep):
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        
        # Check if trajectory exists
        traj_name = f'traj_{traj_idx}'
        if traj_name not in data_group:
            print(f"Trajectory {traj_name} not found in dataset")
            return
            
        traj_group = data_group[traj_name]
        num_samples = traj_group.attrs['num_samples']
        
        if timestep >= num_samples:
            print(f"Timestep {timestep} out of range. Trajectory has {num_samples} timesteps")
            return
            
        # Get images from specified timestep
        image = traj_group['image'][timestep]
        wrist_image = traj_group['wrist_image'][timestep]
        
        # Display images
        cv2.imshow('Main Camera', image)
        cv2.imshow('Wrist Camera', wrist_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_idx', type=int, required=True, 
                        help='Index of trajectory to view')
    parser.add_argument('--timestep', type=int, required=True,
                        help='Timestep to view')
    args = parser.parse_args()
    
    hdf5_path = '/home/weirdlab/ur_bc/data_peg_insertion/raw_demonstrations.h5'
    view_trajectory_images(hdf5_path, args.traj_idx, args.timestep) 