import numpy as np

data_filepath = "/home/weirdlab/ur_bc/data/traj_0.npz"
data = dict(np.load(data_filepath, allow_pickle=True).items())

def isZeroAction(action_list):
    return not np.any(action_list)

def sort_numeric_strings(string_list):
    return sorted(string_list, key=int)

# Example usage
numeric_strings = ["10", "2", "33", "21", "5"]
sorted_list = sort_numeric_strings(numeric_strings)

# Get 0 actions
traj_len = len(data)
num_zero_actions = 0
print(traj_len)
keys = list(data.keys())
sorted_keys = sort_numeric_strings(keys)

for t in sorted_keys:
    obs, action = data[str(t)]
    if isZeroAction(action['left_arm']) and isZeroAction(action['right_arm']):
        print("Found zero action at timestep", t)
        num_zero_actions += 1
        data.pop(str(t), None)

print("num_zero_actions", num_zero_actions)
