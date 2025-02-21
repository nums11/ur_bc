import urx
from time import sleep

"""
Right arm: 2.2
Left arm: 1.2
"""

# right_arm_ip = "192.168.2.2"
left_arm_ip = "192.168.1.2"

# right_arm = urx.Robot(right_arm_ip)
left_arm = urx.Robot(left_arm_ip, use_rt=True)

# print("Right arm joint positions:", right_arm.getj())
while True:
    print("Left arm force:", left_arm.get_force())
    sleep(0.1)
