# from pymodbus.client import ModbusSerialClient as ModbusClient
# import time

# # Initialize the Modbus client
# client = ModbusClient(
#     method='rtu', 
#     port='/dev/ttyUSB2',  # Replace with your USB device
#     baudrate=115200,
#     stopbits=1,
#     parity='N',
#     bytesize=8,
#     timeout=1
# )

# if not client.connect():
#     print("Failed to connect to the gripper. Check your setup!")
#     exit()

# # Function to activate the gripper
# def activate_gripper():
#     # Write to the activation register (register 0x03E8, address 1000)
#     client.write_register(0x03E8, 0x0001, unit=9)  # Unit ID 9 is the default
#     time.sleep(2)  # Wait for activation

# # Function to set the gripper position
# def move_gripper(position):
#     # Write to the position register (0x03E9, address 1001)
#     # Position ranges from 0 (open) to 255 (fully closed)
#     client.write_register(0x03E9, position, unit=9)
#     time.sleep(1)

# # Activate and move the gripper
# try:
#     activate_gripper()
#     print("Gripper activated!")

#     print("Closing gripper...")
#     move_gripper(255)  # Fully close

#     print("Opening gripper...")
#     move_gripper(0)  # Fully open

# except Exception as e:
#     print(f"An error occurred: {e}")

# finally:
#     client.close()
from time import sleep


from pyRobotiqGripper import RobotiqGripper

gripper = RobotiqGripper(portname='/dev/ttyUSB0')

gripper.activate()
gripper.calibrate(0, 40)
print("Calibrated")
sleep(2)

gripper.open()
print("Opened")
sleep(2)

gripper.close()
print("Closed")
sleep(2)

# gripper.goTo(100)
# print("Go to 100")
# sleep(2)

# position_in_bit = gripper.getPosition()
# gripper.goTomm(25)
# print("Go to 25")
# sleep(2)

position_in_mm = gripper.getPositionmm()
print("position_in_mm", position_in_mm)

gripper.printInfo()