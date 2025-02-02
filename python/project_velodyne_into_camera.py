import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import open3d as o3d
import argparse

from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel
from interpolate_poses import interpolate_vo_poses
from velodyne import load_velodyne_raw, velodyne_raw_to_pointcloud


EXTRINSICS = '/home/carlo/Documents/tesi/server/robotcar-dataset-sdk/extrinsics'
MODELS_DIR = '/home/carlo/Documents/tesi/server/robotcar-dataset-sdk/models'

POSES_FILE = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/vo/vo.csv'
STEREO_LEFT = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/stereo/left'
VELO_L = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left'
VELO_R = '/home/carlo/Downloads/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right'
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
            # Elimina i punti fino a 2.7 metri di distanza e con il valore della x positivo (dietro la macchina)
            if d > 2.7 and scan[0, j] < 0:
                tmp.append(scan[:,j])
        tmp = np.array(tmp)
        tmp = np.transpose(tmp)
        scan = tmp

        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))

        if interval == None:
            pointcloud = np.hstack([pointcloud, scan])
        elif i % interval == 0:
            pointcloud = np.hstack([pointcloud, scan])


    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud, reflectance


def single_time():
    timestamp = int(IMAGE)

    model = CameraModel(MODELS_DIR, STEREO_LEFT)

    extrinsics_path = os.path.join(EXTRINSICS, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)

    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

    pointcloud, reflectance = bpc(lidar_dir=VELO_L, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                  start_time=timestamp, end_time=timestamp + 1e5, origin_time=timestamp)
    pointcloud_l = np.dot(G_camera_posesource, pointcloud)
    print(pointcloud_l.shape)

    pointcloud, reflectance = bpc(lidar_dir=VELO_R, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                  start_time=timestamp, end_time=timestamp + 1e5, origin_time=timestamp)
    pointcloud_r = np.dot(G_camera_posesource, pointcloud)
    print(pointcloud_r.shape)

    pointcloud = np.array([[0], [0], [0], [0]])
    pointcloud = np.hstack([pointcloud, pointcloud_l])
    pointcloud = np.hstack([pointcloud, pointcloud_r])

    image_path = os.path.join(STEREO_LEFT, str(timestamp) + '.png')
    image = load_image(image_path, model)

    print(pointcloud.shape)

    uv, depth = model.project(pointcloud, image.shape)

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
    parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
    parser.add_argument('--mode', type=str, default='single')
    args = parser.parse_args()
    if args.mode == 'single':
        single_time()
    else:
        multi_time()




def lms_depth_evaluation():
    print('LMS depth evaluation')

    test_file = open(TEST_FILE, 'r')
    lines = test_file.readlines()
    test_file.close()

    match_file = open(MATCH_FILE, 'r')
    match_lines = match_file.readlines()
    match_file.close()

    timestamps = []

    for line in lines:
        basename = os.path.basename(line.rstrip()).split('.')[0]
        for match_line in match_lines:
            match_line = match_line.rstrip().split(' ')
            if int(match_line[0]) == int(basename):
                timestamps.append(int(match_line[1]))

    model = CameraModel(MODELS_DIR, STEREO_LEFT)

    extrinsics_path = os.path.join(EXTRINSICS, model.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)

    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

    errors = []

    for i in range(0, len(timestamps)):
        timestamp = timestamps[i]
        prediction = os.path.basename(lines[i].rstrip()).split('.')[0]

        print(timestamp, prediction)

        pointcloud, reflectance = build_pointcloud(LMS_FRONT, POSES_FILE, EXTRINSICS,
                                                   timestamp - 1e7, timestamp + 1e7, timestamp)

        pointcloud = np.dot(G_camera_posesource, pointcloud)

        image_path = os.path.join(STEREO_LEFT, str(timestamp) + '.png')

        image = load_image(image_path, model)

        uv, depth = model.project(pointcloud, image.shape)

        #print('image shape {}, uv shape {}, len depth {}'.format(image.shape, uv.shape, len(depth)))

        # 0 < uv[0, :] < 1280
        # 0 < uv[1, :] < 960
        b = []
        d = []
        count = 0
        for i in range(0, len(depth)):
            if CROP_AREA[1] < uv[1, i] < CROP_AREA[3] and  MIN_DEPTH < depth[i] < MAX_DEPTH:
                count += 1
                b.append([uv[0, i], uv[1, i]])
                d.append(depth[i])

        a = np.ndarray((2, count))

        for i in range(0, count):
            a[0, i] = b[i][0]
            a[1, i] = b[i][1] - CROP_AREA[1]

        img = image[CROP_AREA[1]:CROP_AREA[3], CROP_AREA[0]:CROP_AREA[2], :]

        pred_disp = np.load(PREDICTIONS_DIR + '/' + '{}_disp.npy'.format(prediction))

        # (1, WIDTH, HEIGHT)
        pred_disp = pred_disp[0]
        # (WIDTH, HEIGHT)
        pred_disp = pred_disp[0]
        #print(pred_disp.shape)

        pred_disp = cv2.resize(pred_disp, (img.shape[1], img.shape[0]))
        pred_depth = 1 / pred_disp
        #print(pred_depth.shape)

        depth_mask = np.zeros((pred_depth.shape[0], pred_depth.shape[1]))
        #print(depth_mask.shape)

        for i in range(0, a.shape[1]):
            if depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] == 0:
                depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] = d[i]

            elif depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] > d[i]:
                depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] = d[i]

        #print(depth_mask.shape)

        dm = np.logical_and(depth_mask > MIN_DEPTH, depth_mask < MAX_DEPTH)

        dtpm = depth_mask[dm]
        prdpth = pred_depth[dm]

        #print(len(depth_mask[dm]))
        #print(len(pred_depth[dm]))

        ratio = np.median(dtpm) / np.median(prdpth)
        #print(ratio)
        prdpth *= ratio

        e = compute_errors(dtpm, prdpth)
        errors.append(e)



    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == '__main__':
    main()
