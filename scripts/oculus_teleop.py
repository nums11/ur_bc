from OculusTeleopInterface import OculusTeleopInterface
from BimanualUREnv import BimanualUREnv
from time import sleep, time
import numpy as np

oculus_teleop = OculusTeleopInterface()
oculus_teleop.startTeleop()

env = BimanualUREnv(reset_arms=True)
env.reset()

while True:
    left_controller_delta, right_controller_delta, \
        left_trigger, right_trigger = oculus_teleop.getDeltasAndTriggers()
    
    left_arm_action = np.append(left_controller_delta, left_trigger)
    right_arm_action = np.append(right_controller_delta, right_trigger)
    action = left_arm_action, right_arm_action

    print("Stepping with action", action)
    # start = time()
    env.step(action, convert_oculus_deltas=True)
    # end = time()
    # print("\nStep function took", end-start, "seconds", "\n")

    # sleep(0.005)
