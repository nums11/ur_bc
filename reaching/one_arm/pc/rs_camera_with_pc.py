import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

# Create point cloud object
pc = rs.pointcloud()

# Align depth frame to color frame
align = rs.align(rs.stream.color)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Visualize images
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_image)

        color_image = color_image.astype(np.float32) / 255
        print("color_image", color_image)

        # Get point cloud data
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        # Save point cloud data
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(vertices)
        print("Shape", color_image.shape)
        point_cloud.colors = o3d.utility.Vector3dVector(color_image.reshape(-1,3))

        # Press 's' to save the point cloud
        if cv2.waitKey(1) & 0xFF == ord('s'):
            o3d.io.write_point_cloud("output_pointcloud.ply", point_cloud)
            print("Point cloud saved as 'output_pointcloud.ply'")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
