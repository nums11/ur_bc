import os
import numpy as np
import time
import h5py
from tqdm import tqdm
import cv2

# Update path to your trajectory file (adjust as needed)
# dataset_path = '/home/weirdlab/ur_bc/data/raw_demonstrations.h5'
dataset_path = '/home/weirdlab/drawer_demos/data/raw_demonstrations.h5'
traj_str = 'traj_35'
timestep = 100

# Open the HDF5 file
with h5py.File(dataset_path, 'r') as f:
    # Assuming we want to look at the first image in the trajectory
    # The images should be stored in the 'obs' group
    img = f['data'][traj_str]['image'][timestep]  # Get the first image
    wrist_img = f['data'][traj_str]['wrist_image'][timestep]  # Get the first image
    
    # Convert the image to a format suitable for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow('Image', img)
    cv2.imshow('Wrist Image', wrist_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()