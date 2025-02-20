import os
import numpy as np
import time
import h5py
from tqdm import tqdm
import cv2

traj_path='/home/weirdlab/ur_bc/data/traj_0.npz'
traj = dict(np.load(traj_path, allow_pickle=True).items())


img = traj['310'][0]['image']
# Convert the image to a format suitable for OpenCV
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow('Trajectory Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()