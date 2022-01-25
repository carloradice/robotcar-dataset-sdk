import glob

from python.velodyne import *
import open3d as o3d
import numpy as np
from python.transform import build_se3_transform
import glob

VELO_L = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left/1547131046310775.png'
VELO_R = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right/1547131046328585.png'

l = []
r = []
for f in glob.glob('/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left/*.png'):
    l.append(int(os.path.basename(f).split('.')[0]))

for f in glob.glob('/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right/*.png'):
    r.append(int(os.path.basename(f).split('.')[0]))

min = 20000000
for i in range(0, len(l)):
    for j in range(0, len(l)):
        if abs(l[i] - r[j]) < min:
            min = abs(l[i] - r[j])
            #print(l[i], r[j], abs(l[i] - r[j]), i)

pcds = []

ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(VELO_L)
ptcld_0 = velodyne_raw_to_pointcloud(ranges, intensities, angles)
pcds.append(ptcld_0)

print(ptcld_0.shape)

ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(VELO_R)
ptcld_1 = velodyne_raw_to_pointcloud(ranges, intensities, angles)
pcds.append(ptcld_1)


vis = o3d.Visualizer()
vis.create_window()

pcd_0 = o3d.geometry.PointCloud()
# initialise the geometry pre loop
pcd_0.points = o3d.utility.Vector3dVector(ptcld_0[:3].transpose().astype(np.float64))
pcd_0.colors = o3d.utility.Vector3dVector(np.tile(ptcld_0[3:].transpose(), (1, 3)).astype(np.float64))
# Rotate pointcloud to align displayed coordinate frame colouring
pcd_0.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
vis.add_geometry(pcd_0)
render_option = vis.get_render_option()
render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
render_option.point_color_option = o3d.PointColorOption.ZCoordinate
coordinate_frame = o3d.geometry.create_mesh_coordinate_frame()
vis.add_geometry(coordinate_frame)
view_control = vis.get_view_control()
params = view_control.convert_to_pinhole_camera_parameters()
params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
view_control.convert_from_pinhole_camera_parameters(params)

pcd_0.points = o3d.utility.Vector3dVector(ptcld_0[:3].transpose().astype(np.float64))
pcd_0.colors = o3d.utility.Vector3dVector(
    np.tile(ptcld_0[3:].transpose(), (1, 3)).astype(np.float64) / 40)
vis.add_geometry(pcd_0)
vis.update_geometry()
vis.poll_events()
vis.update_renderer()

pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(ptcld_1[:3].transpose().astype(np.float64))
pcd_1.colors = o3d.utility.Vector3dVector(
    np.tile(ptcld_1[3:].transpose(), (1, 3)).astype(np.float64) / 40)
vis.add_geometry(pcd_1)
vis.update_geometry()
vis.poll_events()
vis.update_renderer()

vis.run()
