import numpy as np
from matplotlib import pyplot as plt
import os

# def getActionSemantics(action):
#     if action == 0:
#         return "Move Forward"
#     elif action == 1:
#         return "Move Backward"
#     elif action == 2:
#         return "Move Left"
#     elif action == 3:
#         return "Move Right"
#     elif action == 4:
#         return "Move Up"
#     elif action == 5:
#         return "Move Down"

traj_idx = 7
# sample_idx = 26
# filename = './data/traj_' + str(traj_idx) + '/sample_' + str(sample_idx) +'.npy'

# sample = np.load(filename, allow_pickle=True)
# obs, action = sample
# print("obs:", obs)
# print("action:", action)

data_dir = './data/traj_' + str(traj_idx)
num_samples = len(os.listdir(data_dir))
for i in range(num_samples):
    sample_path = data_dir + '/sample_' + str(i) + '.npy'
    sample = np.load(sample_path, allow_pickle=True)
    obs, action = sample
    if obs[-1] == 1.0:
        print("Gripper was closed")

print("done")