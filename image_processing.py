import argparse
import multiprocessing
import os
import matplotlib
import re
from shutil import copy
from multiprocessing import Pool
import pandas as pd

from python.image import load_image
from python.camera_model import *

models_dir = '/home/radice/neuralNetworks/robotcar-dataset-sdk/models'

def parse_args():
    parser = argparse.ArgumentParser(
        description='processing oxford raw images')

    parser.add_argument('--data_path', type=str,
                        help='path to an image or folder of images', required=True)

    parser.add_argument('--save_path', type=str,
                        help='path where to save processed images', required=True)

    return parser.parse_args()


def loop_function(tuple):
    index = tuple[0]
    file = tuple[1]
    file_path = os.path.join(data_path, global_dir, '{}{}'.format(str(file), '.png'))
    processed = load_image(file_path, camera_model)
    save_file_path = os.path.join(save_path, global_dir, '{}{}'.format(index, '.jpg'))
    matplotlib.image.imsave(save_file_path, processed)


def process(models_dir, data_path, dir, save_path, file_list_int):
    save_data_path = os.path.join(save_path, dir)

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    data_dir = os.path.join(data_path, dir)

    global camera_model
    camera_model = CameraModel(models_dir, data_dir)

    print('IMAGES SAVED AT: ', save_data_path)

    #cpu_count = multiprocessing.cpu_count()
    # usa n core
    pool = multiprocessing.Pool(4)
    pool.map(loop_function, enumerate(file_list_int))

    if dir == 'left':
        # list of tuples
        association_list = []
        association_file_path = os.path.join('/home/radice/neuralNetworks', '{}{}'.format('oxford_association', '.csv'))
        for index, file in enumerate(file_list_int):
            row = []
            row.append(int(file))
            row.append(int(index))
            association_list.append(row)
        df = pd.DataFrame(association_list, columns=['timestamp', 'frame_number'])
        print(df)
        df.to_csv(association_file_path, index=False)


def image_processing(args):
    global data_path
    data_path = args.data_path
    global save_path
    save_path = args.save_path
    # Image Demosaicing and Undistortion
    # se il percorso è un file
    if os.path.isfile(data_path):
        camera_model = CameraModel(models_dir, data_path)
        image = load_image(data_path, camera_model)
        #matplotlib.image.imsave(save_path, image)
    # se il percorso è una cartella
    if os.path.isdir(data_path):
        sub = os.listdir(data_path)

        left_folder = 'left'
        right_folder = 'right'

        data_path_left = os.path.join(data_path, left_folder)
        data_path_right = os.path.join(data_path, right_folder)

        right_file_list = [file for file in os.listdir(data_path_right) if not file.startswith('.')]
        right_file_list_int = [(int(e.split('.')[0])) for e in right_file_list]
        right_file_list_int = sorted(right_file_list_int)

        left_file_list = [file for file in os.listdir(data_path_left) if not file.startswith('.')]
        left_file_list_int = [(int(e.split('.')[0])) for e in left_file_list]
        left_file_list_int = sorted(left_file_list_int)

        for idx in range(0, len(right_file_list_int)):
            if not idx + 1 == len(right_file_list_int):
                if ((right_file_list_int[idx + 1] - right_file_list_int[idx]) < 0):
                    raise Exception("LIST NOT SORTED")

        for idx in range(0, len(left_file_list_int)):
            if not idx + 1 == len(left_file_list_int):
                if ((left_file_list_int[idx + 1] - left_file_list_int[idx]) < 0):
                    raise Exception("LIST NOT SORTED")

        if not len(right_file_list) == len(left_file_list):
            raise Exception(
                "Not same number of images: right={}, left={}".format(len(right_file_list), len(left_file_list)))

        global global_dir

        for dir in sub:
            if dir == 'left':
                global_dir = dir
                process(models_dir, data_path, dir, save_path, left_file_list_int)
            if dir == 'right':
                global_dir = dir
                process(models_dir, data_path, dir, save_path, right_file_list_int)


if __name__ == "__main__":
    args = parse_args()
    image_processing(args)