import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from PIL import Image

from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel
from project_velodyne_into_camera import bpc
from build_pointcloud import build_pointcloud

CROP_AREA = [0, 200, 1280, 810]
MIN_DEPTH = 1e-3
MAX_DEPTH = 80

EXTRINSICS = '/home/carlo/Documents/tesi/server/robotcar-dataset-sdk/extrinsics'
MODELS_DIR = '/home/carlo/Documents/tesi/server/robotcar-dataset-sdk/models'

# Hard coded
PREDICTIONS_DIR = '/home/carlo/Documents/datasets/oxford-radar/md2-oxford-mono-640x416/2019-01-10-14-36-48-radar-oxford-10k-partial/stereo/left'
STEREO_LEFT = '/home/carlo/Documents/datasets/oxford-radar/2019-01-10-14-36-48-radar-oxford-10k-partial/stereo/left'
VELO_L = '/home/carlo/Documents/datasets/oxford-radar/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left'
VELO_R = '/home/carlo/Documents/datasets/oxford-radar/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_right'
LMS_FRONT = '/home/carlo/Documents/datasets/oxford-radar/2019-01-10-14-36-48-radar-oxford-10k-partial/lms_front'
POSES_FILE = '/home/carlo/Documents/datasets/oxford-radar/2019-01-10-14-36-48-radar-oxford-10k-partial/vo/vo.csv'
TEST_FILE = '/home/carlo/Documents/datasets/oxford-radar/oxford_radar_large_test_files.txt'
MATCH_FILE = '/home/carlo/Documents/datasets/oxford-radar/2019-01-10-14-36-48-radar-oxford-10k-partial/match.csv'
POST_PROCESSED = '/home/carlo/Documents/datasets/oxford-radar/oxford-mono-640x416_post_processed_disps_eigen_split.npy'
####


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))

    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def velodyne_depth_evaluation():
    print('Velodyne depth evaluation')

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

    l = []
    for f in sorted(glob.glob(VELO_L + '/*.png')):
        l.append(int(os.path.basename(f).split('.')[0]))

    errors = []

    for i in range(0, len(timestamps)):
        timestamp = timestamps[i]
        prediction = os.path.basename(lines[i].rstrip()).split('.')[0]

        # prendo la scansione velodyne successiva più vicina al frame
        n = 10000000000000000000
        for v in l:
            if v > timestamp and v < n:
                n = v

        #print('Next timestamp{}'.format(n))
        pointcloud, reflectance = bpc(lidar_dir=VELO_L, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                      start_time=n, end_time=n, origin_time=-1)
        pointcloud_l = np.dot(G_camera_posesource, pointcloud)

        # print(pointcloud_l.shape)

        r = []
        for f in sorted(glob.glob(VELO_R + '/*.png')):
            r.append(int(os.path.basename(f).split('.')[0]))
        # prendo la scansione velodyne successiva più vicina al frame
        n = 10000000000000000000
        for v in r:
            if v > timestamp and v < n:
                n = v
        #print('Next timestamp{}'.format(n))
        pointcloud, reflectance = bpc(lidar_dir=VELO_R, extrinsics_dir=EXTRINSICS, poses_file=POSES_FILE,
                                      start_time=n, end_time=n, origin_time=-1)

        pointcloud_r = np.dot(G_camera_posesource, pointcloud)

        #print(pointcloud_r.shape)

        pointcloud = np.hstack([pointcloud_l, pointcloud_r])

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
                depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] = depth[i]

            elif depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] > depth[i]:
                depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] = depth[i]

        #print(depth_mask.shape)

        dm = np.logical_and(depth_mask > MIN_DEPTH, depth_mask < MAX_DEPTH)

        dtpm = depth_mask[dm]
        prdpth = pred_depth[dm]

        #print(len(depth_mask[dm]))
        #print(len(pred_depth[dm]))

        ratio = np.median(dtpm) / np.median(prdpth)
        #print(ratio)
        prdpth *= ratio

        errors.append(compute_errors(dtpm, prdpth))

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


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
                depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] = depth[i]

            elif depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] > depth[i]:
                depth_mask[int(np.floor(a[1, i])), int(np.floor(a[0, i]))] = depth[i]

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
    # velodyne_depth_evaluation()

    lms_depth_evaluation()
