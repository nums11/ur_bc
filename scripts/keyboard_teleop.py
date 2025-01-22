# from ContKeyboardTeleopInterface import ContKeyboardTeleopInterface

# teleop = ContKeyboardTeleopInterface()
# teleop.startTeleop()
import sys
import os
# Add the root directory of the project to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.KeyboardTeleopInterface import KeyboardTeleopInterface

teleop = KeyboardTeleopInterface(use_camera=False)
teleop.start()

