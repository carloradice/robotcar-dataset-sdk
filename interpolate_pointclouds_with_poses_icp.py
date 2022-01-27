
from python.velodyne import *
import open3d as o3d
import numpy as np
import glob
import copy
from python.transform import build_se3_transform


VELO_L = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left'
VELO_R = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right'


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
    #draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    print('Source points {},\ntarget points {}'.format(source.points, target.points))
    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
    print(evaluation)
    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(), o3d.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    #draw_registration_result(source, target, reg_p2p.transformation)

    source.transform(reg_p2p.transformation)

    return source + target


def get_pcl(path):
    ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(path)
    pcl = velodyne_raw_to_pointcloud(ranges, intensities, angles)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:3].transpose().astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(pcl[3:].transpose(), (1, 3)).astype(np.float64) / 40)
    pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))

    if 'velodyne_right' in path:
        pcd.translate((0, -0.71, 0))

    return pcd


def main():
    l = []
    r = []
    for f in sorted(glob.glob('/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left/*.png')):
        l.append(int(os.path.basename(f).split('.')[0]))

    for f in sorted(glob.glob('/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right/*.png')):
        r.append(int(os.path.basename(f).split('.')[0]))

    for i in range(0, 1):
        print(VELO_L, '{}.png'.format(l[i]))
        print(VELO_R, '{}.png'.format(r[i]))
        pcd_l = get_pcl(os.path.join(VELO_L, '{}.png'.format(l[i])))
        pcd_r = get_pcl(os.path.join(VELO_R, '{}.png'.format(r[i])))
        m1 = icp(pcd_l, pcd_r)

        if i > 0:
            m0 = icp(m0, m1)
        else:
            m0 = m1

    o3d.visualization.draw_geometries([m0])


if __name__ == '__main__':
    pcds = main()