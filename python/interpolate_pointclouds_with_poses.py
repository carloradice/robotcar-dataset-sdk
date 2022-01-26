
from velodyne import *
import open3d as o3d
import numpy as np
import glob
import copy
from build_pointcloud import build_pointcloud
from transform import build_se3_transform


VELO_L = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left'
VELO_R = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right'
EXTRINSICS = '/home/carlo/Documents/tesi/server/robotcar-dataset-sdk/extrinsics'
POSES_FILE = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/vo/vo.csv'


def get_pcl(pcl):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:3].transpose().astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(pcl[3:].transpose(), (1, 3)).astype(np.float64) / 40)
    pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))

    return pcd


def main():
    l = []
    r = []
    for f in sorted(glob.glob('/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left/*.png')):
        l.append(int(os.path.basename(f).split('.')[0]))

    for f in sorted(glob.glob('/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right/*.png')):
        r.append(int(os.path.basename(f).split('.')[0]))

    pcd_l = o3d.geometry.PointCloud()
    tmp_l = o3d.geometry.PointCloud()
    pcd_r = o3d.geometry.PointCloud()
    tmp_r = o3d.geometry.PointCloud()

    #for i in range(0, 10):
    pc, ref = build_pointcloud(lidar_dir=VELO_L, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE, start_time=l[0],
                               end_time=l[len(l)-1], origin_time=-1)
    pc[3,:] = ref
    tmp_l = get_pcl(pc)

    #tmp_l.paint_uniform_color([1, 0.706, 0])

    pcd_l += tmp_l

    pc, ref = build_pointcloud(lidar_dir=VELO_R, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE, start_time=r[0],
                               end_time=r[len(r)-1], origin_time=-1)
    pc[3, :] = ref
    tmp_r = get_pcl(pc)

    #tmp_r.paint_uniform_color([0, 0.651, 0.929])

    pcd_r += tmp_r


    o3d.visualization.draw_geometries([pcd_l, pcd_r])


if __name__ == '__main__':
    pcds = main()