# import socket
# from time import sleep

# robot_ip = "192.168.2.2"  # Replace with your robot's IP address
# port = 29999

# # Connect to the Dashboard Server
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# print("Connecting to robot...")
# s.connect((robot_ip, port))
# response = s.recv(1024).decode('utf-8')
# print("Connected response", response)

# # Load the program
# s.send(b"load /programs/modbus_control.urp\n")
# response = s.recv(1024).decode('utf-8')
# print("Load Response:", response)

# # Start the program
# s.send(b"play\n")
# response = s.recv(1024).decode('utf-8')
# print("Play Response:", response)

# # s.send(b"unlock protective stop\n")
# # response = s.recv(1024).decode('utf-8')
# # print("Unlock Response:", response)   

# # s.send(b"power on\n")
# # response = s.recv(1024).decode('utf-8')
# # print("power on Response:", response)


# # s.send(b"brake release\n")
# # response = s.recv(1024).decode('utf-8')
# # print("Brake release Response:", response)


# # s.send(b"log\n")
# # response = s.recv(1024).decode('utf-8')
# # print("Log Response:", response)


# while True:
#     s.send(b"get loaded program\n")
#     response = s.recv(1024).decode('utf-8')
#     print("Loaded response:", response)
#     s.send(b"running\n")
#     response = s.recv(1024).decode('utf-8')
#     print("Running:", response)
#     # s.send(b"programState\n")
#     # response = s.recv(1024).decode('utf-8')
#     # print("Program State:", response)
#     # s.send(b"robotmode\n")
#     # response = s.recv(1024).decode('utf-8')
#     # print("Robot Mode:", response)
#     # s.send(b"safetymode\n")
#     # response = s.recv(1024).decode('utf-8')
#     # print("Safety Mode:", response)
#     sleep(0.5)


# s.close()


import socket

robot_ip = "192.168.1.2"  # Replace with your robot's IP
port = 30002  # Secondary interface port
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((robot_ip, port))

s =socket.create_connection((robot_ip, port), timeout=0.5)
# response = s.recv(1024).decode('utf-8')
# print("Connected response", response)


# URScript command to move the robot
command = '''global position = [0,0,0,0,0,0]\n
        def convert_port_to_val(port):\n
            val = read_port_register(port)\n
            if val > 32768:\n
                new_val = val - 65536\n
                val = new_val\n
            end\n
            return val\n
        end\n

        def update_position():\n
            position[0] = convert_port_to_val(128)*0.01\n
            position[1] = convert_port_to_val(129)*0.01\n
            position[2] = convert_port_to_val(130)*0.01\n
            position[3] = convert_port_to_val(131)*0.01\n
            position[4] = convert_port_to_val(132)*0.01\n
            position[5] = convert_port_to_val(133)*0.01\n
        end\n

        while(True):\n
            update_position()\n
            allj = get_inverse_kin(p[position[0],position[1],position[2],position[3],position[4],position[5]])\n
            servoj(allj,0.5,0.5,0.5)\n
        end'''


# j = [0.01968411816918251, -1.5463324908770193, 2.1245089029468582, 4.016029281775736, -1.4222207118732229, 2.3174900338107105]
c2 = 'movej([0.03468411816918251, -2.5463324908770193, 2.1245089029468582, 4.016029281775736, -1.4222207118732229, 2.3174900338107105], a=0.1, v=0.05, r=0)'

s.send(c2.encode('utf-8'))
# response = s.recv(1024).decode('utf-8')
# print("Command response", response)


s.close()