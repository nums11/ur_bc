import urx
from time import sleep
from pynput.keyboard import Key, Listener

a = 0.01
v = 0.001

robot = urx.Robot("192.168.1.2")
print("Connected to arm")

robot.movej(tuple([0.18465891986675584, -1.1208277855309028, 1.6412143284563836,
        0.16756103565501818, 0.7728892092394712, 0.03140143760940369]))

def moveArm(action):
    pose = robot.get_pose_array()
    joints = robot.getj()
    print("Joint positions", joints)
    if action == 0:
        pose[0] -= 0.02
    elif action == 1:
        pose[0] += 0.02
    elif action == 2:
        pose[1] += 0.02
    elif action == 3:
        pose[1] -= 0.02
    elif action == 4:
        pose[2] += 0.02
    elif action == 5:
        pose[2] -= 0.02
    robot.movejInvKin(pose)
    # robot.servojInvKin(pose)

    # j = robot.getj()
    # j[0] += 0.2
    # robot.movej(j)


def on_press(key):
    if not hasattr(key, 'char'):
        return

    if key.char == 'w':
        moveArm(0)
    elif key.char == 's':
        moveArm(1)
    elif key.char == 'd':
        moveArm(2)
    elif key.char == 'a':
        moveArm(3)
    elif key.char == 'r':
        moveArm(4)
    elif key.char == 'f':
        moveArm(5)
    elif key.char == 'q':
        return False

print("Enter key for movement: 'w', 'a', 's', 'd'")
with Listener(
        on_press=on_press) as listener:
    listener.join()
