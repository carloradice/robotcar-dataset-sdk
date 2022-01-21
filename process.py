import argparse
import multiprocessing

import glob
import os.path
import timeit, time
from PIL import Image
import random
import csv

from python.image import load_image
from python.camera_model import *

MODELS_DIR = '/home/ubuntu/robotcar-dataset-sdk/models'
MEDIA_PATH = '/home/ubuntu/datasets/oxford'
THRESHOLD = 0.20

def parse_args():
    parser = argparse.ArgumentParser(
        description='processing oxford raw images')

    parser.add_argument('--folder', type=str,
                        help='folder of images of oxford dataset', required=True)

    parser.add_argument('--cores', type=int,
                        help='number of cpu cores for parallelism', default=4)

    return parser.parse_args()


def parallel_processing(tuple):
    """
    Image Demosaicing and Undistortion.
    """
    idx = tuple[0]
    file = tuple[1]

    processed_file = load_image(file, camera_model)

    file_name = str(idx).zfill(10)
    save_processed_file_path = os.path.join(os.path.dirname(file), '{}{}'.format(file_name, '.png'))

    im = Image.fromarray(processed_file)
    im.save(save_processed_file_path)


def main(args):
    """
    Main function.
    """
    # start timer
    # start = timeit.default_timer()

    global camera_model

    folder = args.folder.split(',')[0]
    print('-> Processing {}'.format(folder))

    data_path = os.path.join(MEDIA_PATH, folder, 'stereo')

    if not os.path.isdir(data_path):
        raise Exception ('{} path is not a folder'.format(data_path))

    # Left folder
    file_list = glob.glob(os.path.join(data_path, 'left/*.png'))
    # print('-> Number of frames:', len(file_list))

    camera_model = CameraModel(MODELS_DIR, file_list[0])
    # print(camera_model.camera, camera_model.camera_sensor, camera_model.G_camera_image, camera_model.principal_point)

    lsorted_list = []
    rsorted_list = []
    match_file = open(os.path.join(MEDIA_PATH, folder, 'match.csv'), 'w')
    for idx, file in enumerate(sorted(file_list)):
        lfile = os.path.join(data_path, 'left', os.path.basename(file))
        rfile = os.path.join(data_path, 'right', os.path.basename(file))
        lsorted_list.append((idx, lfile))
        rsorted_list.append((idx, rfile))
        # file con le associazioni nome vecchio e nome nuovo
        match_file.write('{} {}\n'.format(str(idx), os.path.basename(file)))



    N = int(len(lsorted_list) * THRESHOLD)

    # print('-> Sample lenght: {}'.format(N))
    i = random.randint(1, len(lsorted_list)-N-1)
    # print('-> Seq start at', i)

    lsample= lsorted_list[i:i+N]
    rsample = rsorted_list[i:i+N]

    pool = multiprocessing.Pool(args.cores)
    pool.map(parallel_processing, lsample)
    pool.map(parallel_processing, rsample)

    # remove old frames
    for i in range(0, len(lsorted_list)):
        os.remove(lsorted_list[i][1])
        os.remove(rsorted_list[i][1])

    # stop timer
    # stop = timeit.default_timer()

    # total run time
    # total_run_time = int(stop - start)

    # print('-> Images/second:', (N*2)/total_run_time)
    # print('-> Total run time:', time.strftime('%H:%M:%S', time.gmtime(total_run_time)))


if __name__ == "__main__":
    args = parse_args()
    main(args)



