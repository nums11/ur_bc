import h5py
import cv2
import argparse
import numpy as np
import os
import glob

def get_data_dir():
    """Find the data directory relative to the script location"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one directory to find the project root (assuming script is in scripts/)
    project_root = os.path.dirname(script_dir)
    # Default data directory in the project
    data_dir = os.path.join(project_root, 'data')
    return data_dir

def list_available_trajectories(data_dir):
    """List all available trajectory files in the data directory"""
    trajectory_files = glob.glob(os.path.join(data_dir, 'episode_*.hdf5'))
    if not trajectory_files:
        print(f"No trajectory files found in {data_dir}")
        return []
    
    # Sort by episode number
    trajectory_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return trajectory_files

def view_trajectory_images(data_dir, traj_idx, timestep=None):
    """
    View images from a trajectory saved by DataCollectionInterface
    
    Args:
        data_dir: Directory containing the trajectory files
        traj_idx: Index of the trajectory to view
        timestep: Optional timestep to view. If None, creates a slider to browse
    """
    # Construct file path for the requested trajectory
    file_path = os.path.join(data_dir, f'episode_{traj_idx}.hdf5')
    
    if not os.path.exists(file_path):
        print(f"Trajectory file not found: {file_path}")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Get the images data
            camera_images = f['observations/images/camera'][:]
            wrist_images = f['observations/images/wrist_camera'][:]
            
            # Get number of timesteps
            num_timesteps = camera_images.shape[0]
            print(f"Trajectory has {num_timesteps} timesteps")
            
            # If timestep is specified, show only that frame
            if timestep is not None:
                if timestep >= num_timesteps:
                    print(f"Timestep {timestep} out of range")
                    return
                
                camera_image = camera_images[timestep]
                wrist_image = wrist_images[timestep]
                
                cv2.imshow('Main Camera', camera_image)
                cv2.imshow('Wrist Camera', wrist_image)
                print(f"Showing timestep {timestep}. Press any key to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # If no timestep is specified, create a browsable interface
            else:
                current_timestep = 0
                
                def on_trackbar(val):
                    nonlocal current_timestep
                    current_timestep = val
                    # Update displays
                    cv2.imshow('Main Camera', camera_images[current_timestep])
                    cv2.imshow('Wrist Camera', wrist_images[current_timestep])
                
                # Create windows with trackbar
                cv2.namedWindow('Main Camera')
                cv2.namedWindow('Wrist Camera')
                cv2.createTrackbar('Timestep', 'Main Camera', 0, num_timesteps-1, on_trackbar)
                
                # Show initial images
                on_trackbar(0)
                
                print("Use the slider to browse through timesteps. Press 'q' to exit.")
                
                # Main loop for browsing
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('n') or key == ord(']'):  # next frame
                        next_timestep = min(current_timestep + 1, num_timesteps - 1)
                        cv2.setTrackbarPos('Timestep', 'Main Camera', next_timestep)
                    elif key == ord('p') or key == ord('['):  # previous frame
                        prev_timestep = max(current_timestep - 1, 0)
                        cv2.setTrackbarPos('Timestep', 'Main Camera', prev_timestep)
                
                cv2.destroyAllWindows()
                
    except Exception as e:
        print(f"Error opening trajectory file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View images from trajectories collected with DataCollectionInterface')
    parser.add_argument('--traj_idx', type=int, help='Index of trajectory to view')
    parser.add_argument('--timestep', type=int, help='Specific timestep to view (optional)')
    parser.add_argument('--data_dir', type=str, help='Directory containing trajectory data (optional)')
    parser.add_argument('--list', action='store_true', help='List available trajectories')
    
    args = parser.parse_args()
    
    # Get data directory
    data_dir = args.data_dir if args.data_dir else get_data_dir()
    
    # If list flag is set, just list available trajectories and exit
    if args.list:
        trajectory_files = list_available_trajectories(data_dir)
        if trajectory_files:
            print("Available trajectories:")
            for i, file_path in enumerate(trajectory_files):
                traj_idx = int(os.path.basename(file_path).split('_')[1].split('.')[0])
                print(f"  {traj_idx}: {file_path}")
        exit(0)
    
    # Ensure trajectory index is provided if not listing
    if args.traj_idx is None:
        print("Please specify a trajectory index with --traj_idx or use --list to see available trajectories")
        exit(1)
    
    # View the specified trajectory
    view_trajectory_images(data_dir, args.traj_idx, args.timestep) 