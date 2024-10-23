import open3d as o3d
import numpy as np

# Load the point cloud from the .ply file
point_cloud = o3d.io.read_point_cloud("output_pointcloud.ply")
colors = np.asarray(point_cloud.colors)
points = np.asarray(point_cloud.points)

print(colors)
print(colors.shape)
# print(points.shape)



# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud],
                                  window_name="Point Cloud Visualization",
                                  width=800, height=600,
                                  left=50, top=50,
                                  point_show_normal=False)
