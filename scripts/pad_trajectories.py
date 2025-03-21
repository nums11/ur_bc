#!/usr/bin/env python3
import os
import h5py
import json
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Pad all trajectory files to the maximum trajectory length")
    parser.add_argument("--data_dir", default="/media/weirdlab/Windows/real_world_data/",
                        help="Directory containing the trajectory files and metadata")
    parser.add_argument("--metadata_file", default="trajectory_metadata.json",
                        help="Name of the metadata file within the data directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would be done, without actually modifying files")
    return parser.parse_args()

def load_metadata(metadata_path):
    """Load the trajectory metadata from a JSON file."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading metadata file: {e}")
        return None

def get_trajectory_files(data_dir):
    """Get all HDF5 files in the data directory that match the episode_X.hdf5 pattern."""
    trajectory_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith('episode_') and filename.endswith('.hdf5'):
            trajectory_files.append(os.path.join(data_dir, filename))
    return sorted(trajectory_files)

def pad_trajectory(file_path, max_length, dry_run=False):
    """Pad a trajectory file to the maximum length."""
    with h5py.File(file_path, 'r') as f:
        # Check current lengths
        current_length = f['observations/qpos'].shape[0]
        
        # If already at max length, skip
        if current_length >= max_length:
            return False, current_length
        
        # Get data to pad
        qpos = f['observations/qpos'][:]
        camera = f['observations/images/camera'][:]
        wrist_camera = f['observations/images/wrist_camera'][:]
        action = f['action'][:]
        
        # Create padded arrays
        padding_length = max_length - current_length
        
        # Replicate the last timestep for padding
        last_qpos = qpos[-1]
        last_camera = camera[-1]
        last_wrist_camera = wrist_camera[-1]
        last_action = action[-1]
        
        qpos_padded = np.vstack([qpos, np.tile(last_qpos, (padding_length, 1))])
        camera_padded = np.vstack([camera, np.tile(last_camera, (padding_length, 1, 1, 1))])
        wrist_camera_padded = np.vstack([wrist_camera, np.tile(last_wrist_camera, (padding_length, 1, 1, 1))])
        action_padded = np.vstack([action, np.tile(last_action, (padding_length, 1))])
        
        if dry_run:
            print(f"Would pad {file_path} from {current_length} to {max_length} timesteps")
            return False, current_length
    
    # Write padded data back to file
    with h5py.File(file_path, 'r+') as f:
        # Delete existing datasets
        del f['observations/qpos']
        del f['observations/images/camera']
        del f['observations/images/wrist_camera']
        del f['action']
        
        # Create new datasets with padded data
        f.create_dataset('observations/qpos', data=qpos_padded)
        f.create_dataset('observations/images/camera', data=camera_padded, 
                        chunks=(1, 480, 640, 3), dtype='uint8')
        f.create_dataset('observations/images/wrist_camera', data=wrist_camera_padded, 
                        chunks=(1, 480, 640, 3), dtype='uint8')
        f.create_dataset('action', data=action_padded)
    
    return True, current_length

def main():
    args = parse_args()
    
    # Load metadata
    metadata_path = os.path.join(args.data_dir, args.metadata_file)
    metadata = load_metadata(metadata_path)
    
    if not metadata:
        print("Failed to load metadata. Exiting.")
        return
    
    max_length = metadata.get('max_trajectory_length', 0)
    if max_length <= 0:
        print(f"Invalid maximum trajectory length in metadata: {max_length}")
        return
    
    print(f"Maximum trajectory length from metadata: {max_length}")
    
    # Get all trajectory files
    trajectory_files = get_trajectory_files(args.data_dir)
    
    if not trajectory_files:
        print("No trajectory files found.")
        return
    
    print(f"Found {len(trajectory_files)} trajectory files.")
    
    # Pad each trajectory file
    padded_count = 0
    for file_path in tqdm(trajectory_files, desc="Padding trajectories"):
        was_padded, current_length = pad_trajectory(file_path, max_length, args.dry_run)
        if was_padded:
            padded_count += 1
    
    if args.dry_run:
        print(f"Dry run completed. Would pad {padded_count} trajectory files to length {max_length}.")
    else:
        print(f"Padding completed. Padded {padded_count} trajectory files to length {max_length}.")

if __name__ == "__main__":
    main() 