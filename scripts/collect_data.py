from OculusTeleopInterface import OculusTeleopInterface
from time import sleep
import threading
import os
import numpy as np

# Instantiate the Oculus Teleop interface and wait until it's ready
oculus_teleop = OculusTeleopInterface(reset_arms=True)
while not oculus_teleop.isReady():
    continue

# Instantiate global variables
a_pressed_and_released = False
b_pressed_and_released = False
lock = threading.Lock()

# Get the new directory to store the trajectory in based on the 
# current number of stored trajectories
def getDatasetDirectory():
    data_base_dir = '/home/weirdlab/ur_bc/data/'
    num_trajectories = len(os.listdir(data_base_dir))
    dir_name = data_base_dir + 'traj_' + str(num_trajectories) + '/'
    os.mkdir(dir_name)
    return dir_name

# Save a sample from the trajectory as a numpy file
def saveSample(data_dir, timestep, obs, action):
    sample = np.array([obs, action], dtype=object)
    filename = data_dir + 'sample_' + str(timestep)
    np.save(filename, sample)

# Create a new directory for a trajectory and save all the samples
# to a numpy file
def saveTrajectory(trajectory):
    data_dir = getDatasetDirectory()
    print("Saving trajectory to path", data_dir)
    for timestep, (observation, action) in trajectory.items():
        print(timestep, observation, action)
        saveSample(data_dir, timestep, observation, action)
    print("Finished Saving Trajectory -------")

# Thread to constantly listen for buttons from the oculus
# and share their status globally
def buttonListenerThread():
    global a_pressed_and_released, b_pressed_and_released, lock
    a_currently_pressed = False
    b_currently_pressed = False
    a_prev_pressed = False
    b_prev_pressed = False
    print("In button listener thread")
    while True:
        buttons = oculus_teleop.getButtons()
        a_currently_pressed = buttons['A']
        b_currently_pressed = buttons['B']

        if (not a_currently_pressed) and a_prev_pressed:
            with lock:
                a_pressed_and_released = True

        if (not b_currently_pressed) and b_prev_pressed:
            with lock:
                b_pressed_and_released = True

        a_prev_pressed = a_currently_pressed
        b_prev_pressed = b_currently_pressed


def printTrajCollectionMessage():
    print("Press 'A' to start collecting Trajectory, 'A' again to save trajectory, and 'B' to discard trajectory.")

# def listenforAPress():
#     a_pressed = False
#     # Record when a is pressed and return when it is released
#     while True:
#         buttons = oculus_teleop.getButtons()
#         if buttons['A']:
#             a_pressed = True
#         elif not buttons['A'] and a_pressed:
#             return
        
# def listenforAorBPress()

# Thread for collecting trajectory
def collectTrajThread():
    print("In trajectory collection thread")
    global a_pressed_and_released, b_pressed_and_released, lock
    collecting_trajectory = False
    # 'A' must be pressed to start trajectory collection
    collection_freq_hz = 10
    collection_freq_time = 1 / collection_freq_hz
    trajectory = {}
    t = 0
    printTrajCollectionMessage()
    while True:
        if collecting_trajectory:
            # Save a trajectory when 'A' is pressed and released
            # Discard it when 'B' is pressed and released
            if a_pressed_and_released:
                print("Saving Trajectory")
                saveTrajectory(trajectory)
                printTrajCollectionMessage()
            elif b_pressed_and_released:
                print("Discarding Trajectory")
                printTrajCollectionMessage()
            
            # Reset the trajectory variables and button presses on 'A' or 'B' press and release
            if a_pressed_and_released or b_pressed_and_released:
                trajectory = {}
                t = 0
                collecting_trajectory = False
                with lock:
                    a_pressed_and_released = False
                    b_pressed_and_released = False
            else:
                # Add the current obs, action pair to the trajectory 
                obs, action_dict = oculus_teleop.getObsAndAction()
                print("Adding to traj: t", t, "obs", obs, "action_dict", action_dict)
                trajectory[t] = (obs, action_dict)
                t += 1

            # Collect at the specified frequency
            sleep(collection_freq_time)

        else:
            # Start collecting the trajectory when 'A' is pressed and released
            if a_pressed_and_released:
                print("Starting Trajectory Collection")
                collecting_trajectory = True
                with lock:
                    a_pressed_and_released = False
                    b_pressed_and_released = False


    # listenforAPress()
    # print("A was pressed")
    # sleep(0.25)

    # while True:
    #     buttons = oculus_teleop.getButtons()
    #     if buttons['A']:
    #         print("Saving Trajectory")
    #         saveTrajectory(trajectory)
    #     elif buttons['B']:
    #         print("Discarding Trajectory")

    #     if buttons['A'] or buttons['B']:
    #         trajectory = {}
    #         t = 0
    #         a_pressed = False
    #         b_pressed = False
    #         return

# Start threads
button_listener_thread = threading.Thread(target=buttonListenerThread, args=())
collect_traj_thread = threading.Thread(target=collectTrajThread, args=())
button_listener_thread.start()
collect_traj_thread.start()

# collectTrajThread()