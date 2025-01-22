import urx
from time import sleep
from pynput.keyboard import Key, Listener

left_arm = urx.Robot("192.168.1.2")
print("Connected to left arm")
right_arm = urx.Robot("192.168.2.2")
print("Connected to right arm")

def moveArm(action):
    robot = None
    if action in range(6):
        robot = left_arm
    else:
        robot = right_arm
    pose = robot.get_pose_array()
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
    elif action == 6:
        pose[0] += 0.02
    elif action == 7:
        pose[0] -= 0.02
    elif action == 8:
        pose[1] -= 0.02
    elif action == 9:
        pose[1] += 0.02
    elif action == 10:
        pose[2] += 0.02
    elif action == 11:
        pose[2] -= 0.02
    robot.movejInvKin(pose)

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
    elif key.char == 'i':
        moveArm(6)
    elif key.char == 'k':
        moveArm(7)
    elif key.char == 'l':
        moveArm(8)
    elif key.char == 'j':
        moveArm(9)
    elif key.char == 'y':
        moveArm(10)
    elif key.char == 'h':
        moveArm(11)
    elif key.char == 'q':
        return False

print("Enter key for movement:")
print("Left arm: 'w', 'a', 's', 'd', 'r', 'f'")
print("Right arm: 'i', 'j', 'k', 'l', 'y', 'hs'")
with Listener(
        on_press=on_press) as listener:
    listener.join()
