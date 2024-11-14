from time import sleep
from pynput.keyboard import Key, Listener
from robotiq_modbus_controller.driver import RobotiqModbusRtuDriver
import urx

robot = urx.Robot("192.168.2.2")

port = "/dev/ttyUSB0"
gripper = RobotiqModbusRtuDriver(port)
gripper.connect()
gripper.activate()
status = gripper.status()
print("Gripper status", status)

def moveArm(action):
    translation_delta = 0.02
    rotation_delta = 0.1
    gripper_speed = 4
    pose = robot.get_pose_array()
    print("pose", pose)
    joints = robot.getj()
    print("Joint positions", joints)
    if action == 0:
        pose[0] += translation_delta
    elif action == 1:
        pose[0] -= translation_delta
    elif action == 2:
        pose[1] -= translation_delta
    elif action == 3:
        pose[1] += translation_delta
    elif action == 4:
        pose[2] += translation_delta
    elif action == 5:
        pose[2] -= translation_delta
    elif action == 6:
        pose[3] += rotation_delta
    elif action == 7:
        pose[3] -= rotation_delta
    elif action == 8:
        pose[4] += rotation_delta
    elif action == 9:
        pose[4] -= rotation_delta
    elif action == 10:
        pose[5] += rotation_delta
    elif action == 11:
        pose[5] -= rotation_delta
    elif action == 12:
        gripper.move(pos=0, speed=gripper_speed, force=1)
    elif action == 13:
        gripper.move(pos=100, speed=gripper_speed, force=1)
 

    if action in list(range(12)):
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
    elif key.char == 'q':
        moveArm(4)
    elif key.char == 'e':
        moveArm(5)
    elif key.char == 'r': # forward rotation
        moveArm(6)
    elif key.char == 'f': # backward rotation
        moveArm(7)
    elif key.char == 'z': # down-left rotation
        moveArm(8)
    elif key.char == 'x': # up-right rotation
        moveArm(9)
    elif key.char == 'c': # up-left rotation
        moveArm(10)
    elif key.char == 'v': # down-right rotation
        moveArm(11)
    elif key.char == 't':
        moveArm(12)
    elif key.char == 'g':
        moveArm(13)


print("Enter key for movement")
with Listener(
        on_press=on_press) as listener:
    listener.join()
