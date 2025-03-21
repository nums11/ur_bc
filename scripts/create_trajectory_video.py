#!/usr/bin/env python3
import sys
import os
import argparse
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_video_from_trajectory(hdf5_path, output_video_path, fps=30, include_timestep=True, include_joint_info=False):
    """
    Creates a video from trajectory data showing both top and wrist camera images.
    Assumes the new trajectory format and that all images are valid.
    
    Args:
        hdf5_path: Path to the HDF5 file containing trajectory data
        output_video_path: Path to save the output MP4 video
        fps: Frames per second for the output video
        include_timestep: Whether to include timestep counter on frames
        include_joint_info: Whether to include joint positions on frames
    """
    print(f"Creating video from trajectory file: {hdf5_path}")
    print(f"Output video will be saved to: {output_video_path}")
    
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Get number of timesteps
        num_samples = f['observations/qpos'].shape[0]
        print(f"Trajectory has {num_samples} timesteps")
        
        # Get image dimensions from camera dataset
        img_height, img_width, channels = f['observations/images/camera'][0].shape
        print(f"Image dimensions: {img_width}x{img_height}, {channels} channels")
        
        # Calculate the dimensions for the combined video frame
        # We'll place both images side by side
        combined_width = img_width * 2  # Two images side by side
        combined_height = img_height
        
        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        video_writer = cv2.VideoWriter(
            output_video_path, 
            fourcc, 
            fps, 
            (combined_width, combined_height)
        )
        
        # Check if wrist camera images exist
        has_wrist_images = 'observations/images/wrist_camera' in f
        
        # Process each timestep
        print("Processing frames...")
        for t in tqdm(range(num_samples)):
            # Extract camera images - assume they're valid
            top_image = f['observations/images/camera'][t]
            
            # Get wrist image if available, otherwise use black image
            if has_wrist_images:
                wrist_image = f['observations/images/wrist_camera'][t]
            else:
                wrist_image = np.zeros((img_height, img_width, channels), dtype=np.uint8)
            
            # Convert to BGR for OpenCV (OpenCV uses BGR)
            top_image_bgr = cv2.cvtColor(top_image, cv2.COLOR_RGB2BGR)
            wrist_image_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
            
            # Add labels to images
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(top_image_bgr, 'Top Camera', (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(wrist_image_bgr, 'Wrist Camera', (10, 30), font, 1, (0, 255, 0), 2)
            
            # Add timestep if requested
            if include_timestep:
                cv2.putText(
                    top_image_bgr, 
                    f'Timestep: {t}', 
                    (10, img_height - 20), 
                    font, 0.7, (0, 255, 0), 2
                )
            
            # Add joint positions if requested
            if include_joint_info:
                qpos = f['observations/qpos'][t]
                joint_text = f'Joints: {qpos[:-1].round(2)}'
                gripper_text = f'Gripper: {qpos[-1]:.2f}'
                
                # Add joint info to top image
                y_offset = 60
                cv2.putText(
                    top_image_bgr, 
                    joint_text[:30] + ('...' if len(joint_text) > 30 else ''), 
                    (10, y_offset), 
                    font, 0.5, (0, 255, 0), 1
                )
                
                cv2.putText(
                    top_image_bgr, 
                    gripper_text, 
                    (10, y_offset + 20), 
                    font, 0.5, (0, 255, 0), 1
                )
            
            # Combine images side by side
            combined_frame = np.hstack((top_image_bgr, wrist_image_bgr))
            
            # Add frame to video
            video_writer.write(combined_frame)
        
        # Release video writer
        video_writer.release()
        
        print(f"Video created successfully: {output_video_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a video from trajectory data.')
    parser.add_argument('--file', type=str, required=False, 
                        help='Path to the HDF5 dataset file')
    parser.add_argument('--output', type=str, required=False,
                        help='Path to save the output video')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video')
    parser.add_argument('--no-timestep', action='store_false', dest='include_timestep',
                        help='Do not include timestep counter on video frames')
    parser.add_argument('--include-joints', action='store_true', dest='include_joint_info',
                        help='Include joint positions on video frames')
    
    args = parser.parse_args()
    
    # Default file path if none provided
    if args.file is None:
        args.file = os.path.join(os.environ.get('HOME', '/home/nums'), 
                               'projects/ur_bc/data/episode_0.hdf5')
    
    # Default output path if none provided
    if args.output is None:
        # Create output filename based on input filename
        input_base = os.path.basename(args.file)
        input_name = os.path.splitext(input_base)[0]
        args.output = os.path.join(os.path.dirname(args.file), f"{input_name}_video.mp4")
    
    # Create the video
    create_video_from_trajectory(
        args.file, 
        args.output, 
        fps=args.fps,
        include_timestep=args.include_timestep,
        include_joint_info=args.include_joint_info
    )

if __name__ == "__main__":
    main() 