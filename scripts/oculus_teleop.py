from OculusTeleopInterface import OculusTeleopInterface

# Instantiate the Oculus Teleop interface and wait until it's ready
oculus_teleop = OculusTeleopInterface(reset_arms=True)
while not oculus_teleop.isReady():
    continue