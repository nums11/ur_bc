#!/usr/bin/env python3
import os
import h5py
import json
import argparse
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Remove a trajectory file at a specific index")
    parser.add_argument("--data_dir", default="/media/weirdlab/e16676a5-3372-430e-84cd-14e37398d508/data/",
                        help="Directory containing the trajectory files and metadata")
    parser.add_argument("--metadata_file", default="trajectory_metadata.json",
                        help="Name of the metadata file within the data directory")
    parser.add_argument("--idx", type=int, required=True,
                        help="Index of the trajectory to remove")
    parser.add_argument("--reindex", action="store_true",
                        help="Reindex the remaining trajectories to maintain continuous indexing")
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

def save_metadata(metadata_path, metadata):
    """Save metadata to a JSON file."""
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Updated metadata saved to {metadata_path}")

def get_trajectory_length(file_path):
    """Get the length of a trajectory from its HDF5 file."""
    try:
        with h5py.File(file_path, 'r') as f:
            length = f['observations/qpos'].shape[0]
        return length
    except (IOError, KeyError) as e:
        print(f"Error reading trajectory file {file_path}: {e}")
        return 0

def get_max_trajectory_length(data_dir):
    """Calculate the maximum trajectory length from all trajectory files."""
    max_length = 0
    traj_files = [f for f in os.listdir(data_dir) if f.startswith('episode_') and f.endswith('.hdf5')]
    
    for file_name in traj_files:
        file_path = os.path.join(data_dir, file_name)
        length = get_trajectory_length(file_path)
        max_length = max(max_length, length)
    
    return max_length

def remove_trajectory(data_dir, idx, reindex=False, dry_run=False):
    """Remove a trajectory file at the specified index."""
    target_file = os.path.join(data_dir, f'episode_{idx}.hdf5')
    
    if not os.path.exists(target_file):
        print(f"Error: Trajectory file {target_file} does not exist")
        return False, 0
    
    # Get trajectory length for metadata update
    traj_length = get_trajectory_length(target_file)
    
    if dry_run:
        print(f"Would remove trajectory file: {target_file}")
        if reindex:
            print(f"Would reindex all trajectories with index > {idx}")
        return True, traj_length
    
    # Remove the file
    os.remove(target_file)
    print(f"Removed trajectory file: {target_file}")
    
    # Reindex remaining files if requested
    if reindex:
        reindex_trajectories(data_dir, idx)
    
    return True, traj_length

def reindex_trajectories(data_dir, start_idx):
    """Reindex trajectory files after removing one, to maintain continuous indexing."""
    traj_files = sorted([f for f in os.listdir(data_dir) 
                         if f.startswith('episode_') and f.endswith('.hdf5')], 
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Filter files with index greater than the removed one
    files_to_reindex = [f for f in traj_files 
                        if int(f.split('_')[1].split('.')[0]) > start_idx]
    
    print(f"Reindexing {len(files_to_reindex)} trajectory files...")
    
    for file_name in tqdm(files_to_reindex, desc="Reindexing trajectories"):
        current_idx = int(file_name.split('_')[1].split('.')[0])
        new_idx = current_idx - 1
        
        current_path = os.path.join(data_dir, file_name)
        new_path = os.path.join(data_dir, f'episode_{new_idx}.hdf5')
        
        shutil.move(current_path, new_path)
    
    print("Reindexing completed")

def update_metadata(metadata_path, removed_length, reindex=False):
    """Update metadata after removing a trajectory."""
    metadata = load_metadata(metadata_path)
    if not metadata:
        return False
    
    # Update trajectory count and total samples
    metadata['total_trajectories'] -= 1
    metadata['total_samples'] -= removed_length
    
    # Recalculate max trajectory length if needed
    if reindex or removed_length >= metadata['max_trajectory_length']:
        data_dir = os.path.dirname(metadata_path)
        metadata['max_trajectory_length'] = get_max_trajectory_length(data_dir)
    
    return metadata

def main():
    args = parse_args()
    
    # Validate inputs
    if args.idx < 0:
        print("Error: Trajectory index must be non-negative")
        return
    
    metadata_path = os.path.join(args.data_dir, args.metadata_file)
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file {metadata_path} does not exist")
        return
    
    # Load metadata to check if index is valid
    metadata = load_metadata(metadata_path)
    if not metadata:
        return
    
    if args.idx >= metadata['total_trajectories']:
        print(f"Error: Trajectory index {args.idx} is out of range (0-{metadata['total_trajectories']-1})")
        return
    
    # Remove the trajectory
    success, removed_length = remove_trajectory(args.data_dir, args.idx, args.reindex, args.dry_run)
    
    if not success:
        return
    
    # Update metadata
    if not args.dry_run:
        updated_metadata = update_metadata(metadata_path, removed_length, args.reindex)
        if updated_metadata:
            save_metadata(metadata_path, updated_metadata)
            print(f"Updated metadata: {updated_metadata}")
    else:
        print(f"Would update metadata: total_trajectories -> {metadata['total_trajectories']-1}, total_samples -> {metadata['total_samples']-removed_length}")
        if args.reindex or removed_length >= metadata['max_trajectory_length']:
            print(f"Would recalculate max_trajectory_length")

if __name__ == "__main__":
    main() 