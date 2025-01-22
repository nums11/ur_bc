import numpy as np
from matplotlib import pyplot as plt

def getActionSemantics(action):
    if action == 0:
        return "Move Forward"
    elif action == 1:
        return "Move Backward"
    elif action == 2:
        return "Move Left"
    elif action == 3:
        return "Move Right"
    elif action == 4:
        return "Move Up"
    elif action == 5:
        return "Move Down"

traj_idx = 6
sample_idx = 28
filename = './data/traj_' + str(traj_idx) + '/sample_' + str(sample_idx) +'.npy'

sample = np.load(filename, allow_pickle=True)
obs, action = sample
print("obs:", obs)
print("action:", action)