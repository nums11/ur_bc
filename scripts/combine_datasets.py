import h5py
import os
import argparse
import numpy as np

def combine_hdf5_datasets(file1_path, file2_path, output_path):
    """
    Combine two HDF5 datasets collected with DataCollectionInterface into a single file.
    
    Args:
        file1_path: Path to the first HDF5 file
        file2_path: Path to the second HDF5 file
        output_path: Path where the combined HDF5 file will be saved
    """
    print(f"Combining datasets:\n  - {file1_path}\n  - {file2_path}\n  -> {output_path}")
    
    # Create output file and data group
    with h5py.File(output_path, 'w') as out_file:
        data_group = out_file.create_group('data')
        data_group.attrs['total_trajectories'] = 0
        data_group.attrs['total_samples'] = 0
        
        # Process first file
        with h5py.File(file1_path, 'r') as file1:
            file1_data = file1['data']
            total_trajectories = file1_data.attrs['total_trajectories']
            total_samples = file1_data.attrs['total_samples']
            
            # Copy each trajectory
            for traj_idx in range(total_trajectories):
                traj_name = f'traj_{traj_idx}'
                src_traj = file1_data[traj_name]
                dst_traj = data_group.create_group(traj_name)
                
                # Copy attributes
                for attr_name, attr_value in src_traj.attrs.items():
                    dst_traj.attrs[attr_name] = attr_value
                
                # Copy datasets
                for key in src_traj.keys():
                    dst_traj.create_dataset(key, data=src_traj[key][()])
            
            # Update metadata
            data_group.attrs['total_trajectories'] = total_trajectories
            data_group.attrs['total_samples'] = total_samples
        
        # Process second file
        with h5py.File(file2_path, 'r') as file2:
            file2_data = file2['data']
            file2_total_trajectories = file2_data.attrs['total_trajectories']
            file2_total_samples = file2_data.attrs['total_samples']
            
            # Copy each trajectory with updated indices
            for traj_idx in range(file2_total_trajectories):
                src_traj_name = f'traj_{traj_idx}'
                dst_traj_name = f'traj_{traj_idx + total_trajectories}'
                
                src_traj = file2_data[src_traj_name]
                dst_traj = data_group.create_group(dst_traj_name)
                
                # Copy attributes
                for attr_name, attr_value in src_traj.attrs.items():
                    dst_traj.attrs[attr_name] = attr_value
                
                # Copy datasets
                for key in src_traj.keys():
                    dst_traj.create_dataset(key, data=src_traj[key][()])
            
            # Update metadata
            data_group.attrs['total_trajectories'] += file2_total_trajectories
            data_group.attrs['total_samples'] += file2_total_samples
    
    print(f"Successfully combined datasets!")
    print(f"Total trajectories: {data_group.attrs['total_trajectories']}")
    print(f"Total samples: {data_group.attrs['total_samples']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine two HDF5 datasets collected with DataCollectionInterface')
    parser.add_argument('file1', type=str, help='Path to the first HDF5 file')
    parser.add_argument('file2', type=str, help='Path to the second HDF5 file')
    parser.add_argument('--output', type=str, default='combined_demonstrations.h5',
                        help='Path where the combined HDF5 file will be saved')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.file1):
        raise FileNotFoundError(f"First input file not found: {args.file1}")
    if not os.path.exists(args.file2):
        raise FileNotFoundError(f"Second input file not found: {args.file2}")
    
    # Check if output file already exists
    if os.path.exists(args.output):
        response = input(f"Output file {args.output} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            exit()
    
    combine_hdf5_datasets(args.file1, args.file2, args.output) 