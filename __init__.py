# import sys
# import os

# # Get the current directory
# current_dir = os.path.abspath(os.path.dirname(__file__))

# # Add the current directory to the Python path
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# # Add all subdirectories to the Python path
# for root, dirs, files in os.walk(current_dir):
#     for dir in dirs:
#         dir_path = os.path.join(root, dir)
#         if dir_path not in sys.path:
#             sys.path.append(dir_path)
#             print("Adding ", dir_path)