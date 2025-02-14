import numpy as np

traj_file_path = '/home/weirdlab/ur_bc/data/traj_9.npz'
trajectory = dict(np.load(traj_file_path, allow_pickle=True).items())

# print(trajectory)
# print(True == 1)
# print(True == 0)