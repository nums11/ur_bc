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
    delta_move = 0.02
    gripper_speed = 4
    pose = robot.get_pose_array()
    print("pose", pose)
    joints = robot.getj()
    print("Joint positions", joints)
    if action == 0:
        pose[0] += delta_move
    elif action == 1:
        pose[0] -= delta_move
    elif action == 2:
        pose[1] -= delta_move
    elif action == 3:
        pose[1] += delta_move
    elif action == 4:
        pose[2] += delta_move
    elif action == 5:
        pose[2] -= delta_move
    elif action == 6:
        pose[2] -= delta_move
 

    if action in list(range(6)):
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
    elif key.char == 'r':
        moveArm(6)
    elif key.char == 'f':
        moveArm(7)
    elif key.char == 'z':
        moveArm(8)
    elif key.char == 'x':
        moveArm(9)
    elif key.char == 't':
        moveArm(10)
    elif key.char == 'g':
        moveArm(11)


print("Enter key for movement: 'w', 'a', 's', 'd', 't', 'g'")
with Listener(
        on_press=on_press) as listener:
    listener.join()
