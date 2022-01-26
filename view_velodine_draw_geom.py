
from python.velodyne import *
import open3d as o3d
import numpy as np
import glob
import copy

VELO_L = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left/1547131046310775.png'
VELO_R = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right/1547131046328585.png'

def main():
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

    ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(VELO_R)
    ptcld_1 = velodyne_raw_to_pointcloud(ranges, intensities, angles)


    pcd_0 = o3d.geometry.PointCloud()
    pcd_0.points = o3d.utility.Vector3dVector(ptcld_0[:3].transpose().astype(np.float64))
    pcd_0.colors = o3d.utility.Vector3dVector(
        np.tile(ptcld_0[3:].transpose(), (1, 3)).astype(np.float64) / 40)

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(ptcld_1[:3].transpose().astype(np.float64))
    pcd_1.colors = o3d.utility.Vector3dVector(
        np.tile(ptcld_1[3:].transpose(), (1, 3)).astype(np.float64) / 40)

    pcds.append(pcd_0)

    # 0.71 perchè è sistema di riferimento centrale
    pcd_1.translate((0, -0.71, 0))

    pcds.append(pcd_1)

    #o3d.visualization.draw_geometries(pcds)

    return pcds


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp(source, target):
    # ICP
    threshold = 0.99
    trans_init = np.eye(4)
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
    print(evaluation)
    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(), o3d.ICPConvergenceCriteria(max_iteration = 2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)


if __name__ == '__main__':
    pcds = main()
    icp(source=pcds[0], target=pcds[1])