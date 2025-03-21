from environments.BimanualUREnv import BimanualUREnv
from environments.UREnv import UREnv
from gello_software.gello.agents.gello_agent import GelloAgent
from time import sleep
import threading

class GelloTeleopInterface:
    def __init__(self, env):
        # Initialize the interface
        self.env = env
        assert self.env.action_type == "joint_modbus", "GelloTeleopInterface only supports action_type 'joint_modbus'"
        self.resetting = False
        self.agent = GelloAgent(port='/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9MFSJV-if00-port0')
        self.obs = {}
        print("Initialized GelloTeleopInterface")

    def start(self):
        print("GelloTeleopInterface: Start")
        print("GelloTeleopInterface: MOVE GELLO TO STARTING POSITION BEFORE STARTING UR PROGRAM")
        self.teleop_thread = threading.Thread(target=self._teleopThread)
        self.teleop_thread.start()

    def _teleopThread(self):
        self.reset()
        while True:
            if not self.resetting:
                # Get the servo readings from GELLO
                servo_values = self.agent.act({})

                # Construct action
                action = self._constructActionBasedOnEnv(servo_values)
                
                # Step the environment
                self.obs = self.env.step(action)
                sleep(0.004) # 250hz

    def reset(self):
        self.resetting = True
        self.obs = self.env.reset()
        self.resetting = False

    def _constructActionBasedOnEnv(self, servo_values):
        action = None
        if type(self.env) == BimanualUREnv:
            raise ValueError("BimanualUREnv not supported")
        elif type(self.env) == UREnv:
            joints = servo_values[:6]
            gripper = self._clipGripper(servo_values[6])
            # Construct Action
            action = {
                'arm_j': joints,
                'gripper': gripper
            }
        return action
    
    def _clipGripper(self, gripper):
        return gripper >= 0.5
    
    def getObservation(self):
        return self.obs