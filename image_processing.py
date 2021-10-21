import argparse
import multiprocessing
import os
import matplotlib
import re
from shutil import copy
from multiprocessing import Pool
import pandas as pd
import sys

from python.image import load_image
from python.camera_model import *

models_dir = '/home/radice/neuralNetworks/robotcar-dataset-sdk/models'
media_path = '/media/RAIDONE/radice/'
home_path = '/home/radice/neuralNetworks/splits/OXFORD'

def parse_args():
    parser = argparse.ArgumentParser(
        description='processing oxford raw images')

    parser.add_argument('--data_path', type=str,
                        help='path to a folder of images', required=True)

    parser.add_argument('--save_path', type=str,
                        help='path where to save processed images', required=True)

    parser.add_argument('--cores', type=int,
                        help='number of cpu cores for parallelism', default=1)

    return parser.parse_args()


def parallel_processing(tuple):
    """
    Image Demosaicing and Undistortion.
    """
    index = tuple[0]
    file = tuple[1]
    file_path = os.path.join(data_path, global_dir, '{}{}'.format(str(file), '.png'))
    processed = load_image(file_path, camera_model)
    save_file_path = os.path.join(save_path, global_dir, '{}{}'.format(index, '.jpg'))
    matplotlib.image.imsave(save_file_path, processed)


def image_processing(models_dir, data_path, dir, save_path, file_list_int):
    """
    Executes parallel_processing.
    """
    global camera_model
    save_data_path = os.path.join(save_path, dir)

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
        print('-> PATH', save_data_path, 'CREATED')

    print('-> SAVE PATH: ', save_data_path)

    data_dir = os.path.join(data_path, dir)

    camera_model = CameraModel(models_dir, data_dir)

    # esecuzione preprocessing parallelo
    pool = multiprocessing.Pool(args.cores)
    pool.map(parallel_processing, enumerate(file_list_int))

    # save association file
    if dir == 'left':
        association = []
        splitted = data_path.split('/')
        folder = [s for s in splitted if ('2014' or '2015') in s][0]
        if not os.path.exists(os.path.join(media_path, folder)):
            os.makedirs(os.path.join(media_path, folder))
            print('-> PATH', os.path.join(media_path, folder), 'CREATED')
        file_name = 'frames_association'
        association_file_path = os.path.join(home_path, folder, '{}{}'.format(file_name, '.csv'))
        print('-> ASSOCIATION FILE SAVE PATH:', association_file_path)

        for index, file in enumerate(file_list_int):
            row = []
            row.append(int(file))
            row.append(int(index))
            association.append(row)

        df = pd.DataFrame(association, columns=['timestamp', 'frame_number'])
        df.to_csv(association_file_path, index=False)


def sort(path):
    """
    Sorts a list.
    """
    list = [file for file in os.listdir(path) if not file.startswith('.')]
    list_int = [(int(e.split('.')[0])) for e in list]
    sorted_list = sorted(list_int)

    if not sorted_list:
        raise Exception("LIST NOT SORTED")

    return sorted_list, list_int


def main(args):
    """
    Main function.
    """
    global data_path
    data_path = args.data_path
    global save_path
    save_path = args.save_path
    global global_dir

    if not os.path.isdir(data_path):
        sys.exit('Path is not a folder')

    left_folder = 'left'
    right_folder = 'right'

    data_path_left = os.path.join(data_path, left_folder)
    print('-> LOAD PATH', data_path_left)
    data_path_right = os.path.join(data_path, right_folder)
    print('-> LOAD PATH', data_path_right)

    left_file_list_int, left_file_list = sort(data_path_left)
    right_file_list_int, right_file_list = sort(data_path_right)

    if not len(right_file_list) == len(left_file_list):
        raise Exception(
            'NOT THE SAME NUMBER OF IMAGES: RIGHT={}, LEFT={}'.format(len(right_file_list), len(left_file_list)))

    global_dir = left_folder
    image_processing(models_dir, data_path, left_folder, save_path, left_file_list_int)

    global_dir = right_folder
    image_processing(models_dir, data_path, right_folder, save_path, right_file_list_int)


if __name__ == "__main__":
    args = parse_args()
    main(args)