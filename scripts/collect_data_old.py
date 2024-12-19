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
def getDatasetFilename():
    data_base_dir = '/home/weirdlab/ur_bc/data/'
    num_trajectories = len(os.listdir(data_base_dir))
    filename = data_base_dir + 'traj_' + str(num_trajectories) + '.npz'
    return filename

# Save a sample from the trajectory as a numpy file
def saveSample(data_dir, timestep, obs, action):
    sample = np.array([obs, action], dtype=object)
    filename = data_dir + 'sample_' + str(timestep)
    np.save(filename, sample)

# Create a new directory for a trajectory and save all the samples
# to a numpy file
def saveTrajectory(trajectory):
    filename = getDatasetFilename()
    print("Saving trajectory to path", filename)
    np.savez(filename, **trajectory)
    print("\n --- Finished Saving Trajectory --- \n")

# Thread to constantly listen for buttons from the oculus
# and share their status globally
def buttonListenerThread():
    global a_pressed_and_released, b_pressed_and_released, lock
    a_currently_pressed = False
    b_currently_pressed = False
    a_prev_pressed = False
    b_prev_pressed = False
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
    print("\nPress 'A' to start collecting Trajectory, 'A' again to save trajectory, and 'B' to discard trajectory.\n")

# Thread for collecting trajectory
def collectTrajThread():
    global oculus_teleop, a_pressed_and_released, b_pressed_and_released, lock
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
                saveTrajectory(trajectory)
            elif b_pressed_and_released:
                print("\n --- Discarding Trajectory --- \n")
            
            # Reset the trajectory variables and button presses on 'A' or 'B' press and release
            if a_pressed_and_released or b_pressed_and_released:
                trajectory = {}
                t = 0
                collecting_trajectory = False
                with lock:
                    a_pressed_and_released = False
                    b_pressed_and_released = False
                oculus_teleop.resetArms()
                printTrajCollectionMessage()
            else:
                # Add the current obs, action pair to the trajectory 
                obs, action_dict = oculus_teleop.getObsAndAction()
                print("Adding to traj: t", t, "obs", obs, "action_dict", action_dict)
                trajectory[str(t)] = (obs, action_dict)
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

# Start threads
button_listener_thread = threading.Thread(target=buttonListenerThread, args=())
collect_traj_thread = threading.Thread(target=collectTrajThread, args=())
button_listener_thread.start()
collect_traj_thread.start()