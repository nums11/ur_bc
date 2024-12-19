from OculusTeleopInterface import OculusTeleopInterface
from time import sleep
import threading
import os
import numpy as np

class DataCollectionInterface:
    def __init__(self, collection_freq_hz=10):
        # Initialize member variables
        self.collection_freq_hz = collection_freq_hz
        self.collection_freq_time = 1 / self.collection_freq_hz

        # Instantiate the Oculus Teleop interface and wait until it's ready
        self.oculus_teleop = OculusTeleopInterface(reset_arms=True)
        self.oculus_teleop.startTeleop()
        print()
        print("DataCollectionInterface: Initialized OculusTeleopInterface")

        # Instantiate global variables
        self.a_pressed_and_released = False
        self.b_pressed_and_released = False
        self.lock = threading.Lock()
        print("DataCollectionInterface: Finished Initializing DataCollectionInterface")
        print()

    # Get the new directory to store the trajectory in based on the 
    # current number of stored trajectories
    def getDatasetFilename(self):
        data_base_dir = '/home/weirdlab/ur_bc/data/'
        num_trajectories = len(os.listdir(data_base_dir))
        filename = data_base_dir + 'traj_' + str(num_trajectories) + '.npz'
        return filename

    # Create a new directory for a trajectory and save all the samples
    # to a numpy file
    def saveTrajectory(self, trajectory):
        filename = self.getDatasetFilename()
        print("Saving trajectory to path", filename)
        np.savez(filename, **trajectory)
        print("\n --- Finished Saving Trajectory --- \n")

    def printTrajCollectionMessage(self):
        print("\nPress 'A' to start collecting Trajectory, 'A' again to save trajectory, and 'B' to discard trajectory.\n")

    # Thread to constantly listen for buttons from the oculus
    # and share their status globally
    def buttonListenerThread(self):
        a_currently_pressed = False
        b_currently_pressed = False
        a_prev_pressed = False
        b_prev_pressed = False
        while True:
            buttons = self.oculus_teleop.getButtons()
            a_currently_pressed = buttons['A']
            b_currently_pressed = buttons['B']

            if (not a_currently_pressed) and a_prev_pressed:
                with self.lock:
                    self.a_pressed_and_released = True

            if (not b_currently_pressed) and b_prev_pressed:
                with self.lock:
                    self.b_pressed_and_released = True

            a_prev_pressed = a_currently_pressed
            b_prev_pressed = b_currently_pressed

    # Thread for collecting trajectory
    def collectTrajThread(self):
        collecting_trajectory = False
        trajectory = {}
        t = 0
        self.printTrajCollectionMessage()
        while True:
            if collecting_trajectory:
                # Save a trajectory when 'A' is pressed and released
                # Discard it when 'B' is pressed and released
                if self.a_pressed_and_released:
                    self.saveTrajectory(trajectory)
                elif self.b_pressed_and_released:
                    print("\n --- Discarding Trajectory --- \n")
                
                # Reset the trajectory variables and button presses on 'A' or 'B' press and release
                if self.a_pressed_and_released or self.b_pressed_and_released:
                    trajectory = {}
                    t = 0
                    collecting_trajectory = False
                    with self.lock:
                        self.a_pressed_and_released = False
                        self.b_pressed_and_released = False
                    self.oculus_teleop.resetArms()
                    self.printTrajCollectionMessage()
                else:
                    # Add the current obs, action pair to the trajectory 
                    obs, action_dict = self.oculus_teleop.getObsAndAction()
                    print("Adding to traj: t", t, "obs", obs, "action_dict", action_dict)
                    trajectory[str(t)] = (obs, action_dict)
                    t += 1

                # Collect at the specified frequency
                sleep(self.collection_freq_time)

            else:
                # Start collecting the trajectory when 'A' is pressed and released
                if self.a_pressed_and_released:
                    print("Starting Trajectory Collection")
                    collecting_trajectory = True
                    with self.lock:
                        self.a_pressed_and_released = False
                        self.b_pressed_and_released = False

    """ Start button listener and trajectory collection threads """
    def startDataCollection(self):
        print()
        print("DataCollectionInterface: Begin Data Collection")
        print()
        # Start threads
        button_listener_thread = threading.Thread(target=self.buttonListenerThread, args=())
        collect_traj_thread = threading.Thread(target=self.collectTrajThread, args=())
        button_listener_thread.start()
        collect_traj_thread.start()
