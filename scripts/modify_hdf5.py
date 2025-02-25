import h5py
import argparse

def remove_trajectory(hdf5_path, traj_idx):
    with h5py.File(hdf5_path, 'a') as f:
        data_group = f['data']
        
        # Check if trajectory exists
        traj_name = f'traj_{traj_idx}'
        if traj_name not in data_group:
            print(f"Trajectory {traj_name} not found in dataset")
            return
        
        # Get number of samples in trajectory to update total
        num_samples = data_group[traj_name].attrs['num_samples']
        
        # Delete the trajectory group
        del data_group[traj_name]
        
        # Update metadata
        data_group.attrs['total_trajectories'] -= 1
        data_group.attrs['total_samples'] -= num_samples
        
        print(f"Successfully removed {traj_name} and updated metadata")

def increment_total_trajectories(hdf5_path):
    with h5py.File(hdf5_path, 'a') as f:
        data_group = f['data']
        data_group.attrs['total_trajectories'] += 1
        print(f"Incremented total_trajectories to {data_group.attrs['total_trajectories']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_idx', type=int, help='Index of trajectory to remove')
    parser.add_argument('--increment', action='store_true', 
                        help='Increment total_trajectories by 1')
    args = parser.parse_args()
    
    hdf5_path = '/home/weirdlab/ur_bc/data/raw_demonstrations.h5'
    
    if args.traj_idx is not None:
        remove_trajectory(hdf5_path, args.traj_idx)
    if args.increment:
        increment_total_trajectories(hdf5_path)
