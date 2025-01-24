import rtde_control
# from rtde_control import RTDEControlInterface as RTDEControl

# rtde_frequency = 500.0
# rtde_c = RTDEControl("192.168.2.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
rtde_c = rtde_control.RTDEControlInterface("192.168.1.2")
# rtde_c.moveL([-0.143, -0.435, 0.20, -0.001, 3.12, 0.04], 0.5, 0.3)