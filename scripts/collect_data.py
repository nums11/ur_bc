from OculusTeleopInterface import OculusTeleopInterface

oculus_teleop = OculusTeleopInterface(reset_arms=True)
while not oculus_teleop.isReady():
    continue

while True:
    obs = oculus_teleop.getObservation()
    # print("obs", obs)