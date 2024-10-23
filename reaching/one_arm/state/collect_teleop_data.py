import sys
import os
from pynput import keyboard
from time import sleep
import numpy as np
import urx

robot = urx.Robot("192.168.1.2")
print("Connected to arm. Setting to start position ----")
start_joint__positions = tuple([-0.4877165042104066, -1.593502988132193,
    2.1684843291414904, 0.33130235142425873, 0.23414970368740268, -0.4921767767939169])
robot.movej(start_joint__positions)
print("Robot reset to start position ------")
trajectory = {}
timestep = 0

def moveRobot(pose, action):
    move_increment = 0.02
    if action == 0:
        pose[0] -= move_increment
    elif action == 1:
        pose[0] += move_increment
    elif action == 2:
        pose[1] -= move_increment
    elif action == 3:
        pose[1] += move_increment
    elif action == 4:
        pose[2] += move_increment
    elif action == 5:
        pose[2] -= move_increment
    robot.movejInvKin(pose)

def getValidActions():
    return ['w','s','a','d','r','f']

def getDatasetDirectory():
    data_base_dir = '/home/weirdlab/ur_bc/data/'
    num_trajectories = len(os.listdir(data_base_dir))
    dir_name = data_base_dir + 'traj_' + str(num_trajectories) + '/'
    os.mkdir(dir_name)
    return dir_name

def saveSample(data_dir, timestep, obs, action):
    sample = np.array([obs, action], dtype=object)
    filename = data_dir + 'sample_' + str(timestep)
    np.save(filename, sample)

def saveTrajectory():
    data_dir = getDatasetDirectory()
    print("Saving trajectory to path", data_dir)
    for timestep, (observation, action) in trajectory.items():
        print(timestep, observation, action)
        saveSample(data_dir, timestep, observation, action)
    print("Finished Saving Trajectory -------")

def on_press(key):
    global trajectory, timestep
    valid_actions = getValidActions()
    if hasattr(key, 'char') :
        if key.char == 'x':
            saveTrajectory()
        elif key.char in valid_actions:
            pose = robot.get_pose_array()
            action = valid_actions.index(key.char)
            observation = pose
            moveRobot(pose, action)
            trajectory[timestep] = [observation, action]
            print("t:", timestep, "observation", observation, "action", action)
            timestep += 1
        else:
            print("Invalid action")

print("Keyboard Teleop")
print("x plane: 'w', 's'  y plane: 'a', 'd'   z plane: 'r', 'f'")
with keyboard.Listener(
        on_press=on_press) as listener:
    listener.join()
