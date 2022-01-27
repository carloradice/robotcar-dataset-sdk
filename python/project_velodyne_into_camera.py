import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import open3d as o3d

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud


EXTRINSICS = '/home/carlo/Documents/tesi/server/robotcar-dataset-sdk/extrinsics'
MODELS_DIR = '/home/carlo/Documents/tesi/server/robotcar-dataset-sdk/models'

POSES_FILE = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/vo/vo.csv'
STEREO_LEFT = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/stereo/left'
VELO_L = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left'
VELO_R = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right'
# IMAGE = '1547131048253167'
IMAGE = '1547131046378130'

def bpc(lidar_dir, poses_file, extrinsics_dir, start_time, end_time, origin_time=-1, interval=None):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    lidar = re.search('(velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo)\.csv', poses_file).group(1)

    # sensor is VO, which is located at the main vehicle frame
    poses = interpolate_vo_poses(poses_file, timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    reflectance = np.empty((0))

    for i in range(0, len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
        if not os.path.isfile(scan_path):
            continue
        ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)

        ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

        reflectance = np.concatenate((reflectance, ptcld[3]))
        scan = ptcld[:3]

        tmp = []
        for j in range(0, scan.shape[1]):
            d = np.sqrt(np.power(scan[0, j], 2) + np.power(scan[1, j], 2) + np.power(scan[2, j], 2))
            if d > 2.7 and scan[0, j] < 0:
                tmp.append(scan[:,j])
        tmp = np.array(tmp)
        tmp = np.transpose(tmp)
        scan = tmp


        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))

        # tmp = []
        # for j in range(0, scan.shape[1]):
        #     d = np.sqrt(np.power(scan[0, j], 2) + np.power(scan[1, j], 2) + np.power(scan[2, j], 2))
        #     if d > 1:
        #         tmp.append(scan[:,j])
        #
        # #print(len(tmp))
        # tmp = np.array(tmp)
        # #print(tmp.shape)
        # tmp = tmp[:,:,0]
        # #print(tmp.shape)
        # tmp = np.transpose(tmp)
        # #print(tmp.shape)
        #
        # scan = tmp
        #
        # #print(scan.shape)

        if interval == None:
            pointcloud = np.hstack([pointcloud, scan])
        elif i % interval == 0:
            print(timestamps[i])
            pointcloud = np.hstack([pointcloud, scan])


    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud, reflectance


def single_time():
    model = CameraModel(MODELS_DIR, STEREO_LEFT)

    extrinsics_path = os.path.join(EXTRINSICS, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    poses_type = re.search('(vo|ins|rtk)\.csv', POSES_FILE).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(EXTRINSICS, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    else:
        # VO frame and vehicle frame are the same
        G_camera_posesource = G_camera_vehicle

    timestamp = int(IMAGE)

    l = []
    for f in sorted(glob.glob(VELO_L+'/*.png')):
        l.append(int(os.path.basename(f).split('.')[0]))
    # prendo la scansione velodyne successiva più vicina al frame
    n = 10000000000000000000
    for v in l:
        if v > timestamp and v < n:
            n = v
    print('Next {}'.format(n))
    pointcloud, reflectance = build_pointcloud(lidar_dir=VELO_L, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                               start_time=n, end_time=n, origin_time=-1)
    pointcloud_l = np.dot(G_camera_posesource, pointcloud)
    print(pointcloud_l.shape)

    r = []
    for f in sorted(glob.glob(VELO_R + '/*.png')):
        r.append(int(os.path.basename(f).split('.')[0]))
    # prendo la scansione velodyne successiva più vicina al frame
    n = 10000000000000000000
    for v in r:
        if v > timestamp and v < n:
            n = v
    print('Next {}'.format(n))
    pointcloud, reflectance = build_pointcloud(lidar_dir=VELO_R, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                               start_time=n, end_time=n, origin_time=-1)

    pointcloud_r = np.dot(G_camera_posesource, pointcloud)

    print(pointcloud_r.shape)

    pointcloud = np.hstack([pointcloud_l, pointcloud_r])

    image_path = os.path.join(STEREO_LEFT, str(timestamp) + '.png')
    image = load_image(image_path, model)

    uv, depth = model.project(pointcloud, image.shape)

    plt.imshow(image)
    plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def multi_time():
    model = CameraModel(MODELS_DIR, STEREO_LEFT)

    extrinsics_path = os.path.join(EXTRINSICS, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)

    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

    timestamp = int(IMAGE)

    l = []
    for f in sorted(glob.glob(VELO_L+'/*.png')):
        l.append(int(os.path.basename(f).split('.')[0]))
    # prendo le scansioni velodyne successive
    after = []
    for v in l:
        if v > timestamp:
            after.append(v)
    print('Next {}'.format(after))
    pointcloud, reflectance = bpc(lidar_dir=VELO_L, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                               start_time=after[0], end_time=after[len(after)-1], origin_time=-1,
                                               interval=5)

    pointcloud_l = np.dot(G_camera_posesource, pointcloud)
    print('Left point cloud shape{}'.format(pointcloud_l.shape))

    r = []
    for f in sorted(glob.glob(VELO_R + '/*.png')):
        r.append(int(os.path.basename(f).split('.')[0]))
    # prendo le scansioni velodyne successive
    after = []
    for v in l:
        if v > timestamp:
            after.append(v)
    print('Next {}'.format(after))
    pointcloud, reflectance = bpc(lidar_dir=VELO_R, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                               start_time=after[0], end_time=after[len(after)-1], origin_time=-1,
                                               interval=5)

    pointcloud_r = np.dot(G_camera_posesource, pointcloud)

    print('Right point cloud shape{}'.format(pointcloud_r.shape))

    pointcloud = np.hstack([pointcloud_l, pointcloud_r])

    print('Total point cloud shape{}'.format(pointcloud.shape))

    image_path = os.path.join(STEREO_LEFT, str(timestamp) + '.png')
    image = load_image(image_path, model)

    uv, depth = model.project(pointcloud, image.shape)

    print('min depth {}, max depth {}, depth length {}, points {}'.format(min(depth), max(depth), len(depth), uv.shape))

    plt.imshow(image)
    plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.xticks([])
    plt.yticks([])
    plt.show()


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud[:3].transpose().astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(pointcloud[3:].transpose(), (1, 3)).astype(np.float64) / 40)

    o3d.visualization.draw_geometries([pcd])

def main():
    # Tutto ok
    # single_time()

    # Da sistemare
    multi_time()


if __name__ == '__main__':
    main()
