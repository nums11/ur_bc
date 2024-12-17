import urx

"""
Right arm: 2.2
Left arm: 1.2
"""

right_arm_ip = "192.168.2.2"
left_arm_ip = "192.168.1.2"

right_arm = urx.Robot(right_arm_ip)
left_arm = urx.Robot(left_arm_ip)

print("Right arm joint positions:", right_arm.getj())
print("Left arm joint positions:", left_arm.getj())
