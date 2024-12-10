import open3d as o3d
import numpy as np

# Load the point cloud from a PLY file
pcd = o3d.io.read_point_cloud("output_pointcloud.ply")

points_np = np.asarray(pcd.points)

print(points_np, points_np.shape)

crop_size = 6.0
half_size = crop_size / 2
mask = (np.abs(points_np[:, 0]) <= half_size) & \
        (np.abs(points_np[:, 1]) <= half_size) & \
        (np.abs(points_np[:, 2]) <= half_size)

cropped_points = points_np[mask]

print(cropped_points, cropped_points.shape)

cropped_pcd = o3d.geometry.PointCloud()
cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([cropped_pcd])

# # Define the cropping region around the origin (e.g., a cube with side 2 centered at origin)
# min_bound = np.array([-1.0, -1.0, -1.0])  # Lower corner of the cropping box
# max_bound = np.array([1.0, 1.0, 1.0])    # Upper corner of the cropping box

# # Crop the point cloud based on the bounding box
# bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
# cropped_pcd = pcd.crop(bounding_box)

# # Visualize the cropped point cloud
# o3d.visualization.draw_geometries([cropped_pcd], window_name="Cropped Point Cloud")


# def cropPointCloud(points, crop_size):
#     half_size = crop_size / 2
#     mask = (points[:, 0].abs() <= half_size) & \
#            (points[:, 1].abs() <= half_size) & \
#            (points[:, 2].abs() <= half_size)
    
#     return points[mask]