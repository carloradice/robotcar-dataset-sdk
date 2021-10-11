import argparse
import os
import matplotlib
import re

from python.image import load_image
from python.camera_model import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='processing oxford raw images')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--save_path', type=str,
                        help='path to a test image or folder of images', required=True)

    return parser.parse_args()


def process(models_dir, image_path, dir):
    image_dir = os.path.join(image_path, dir)
    camera_model = CameraModel(models_dir, image_dir)
    print('IMAGES SAVED AT: ', args.save_path + '/' + dir)
    count = 0
    for image_name in os.listdir(image_path + '/' + dir):
        image = load_image(image_path + '/' + dir + '/' + image_name, camera_model)
        image_name = image_name.split('.')[0]
        save_path = os.path.join(args.save_path, dir, '{}.{}'.format(image_name, 'jpg'))
        count += 1
        if count % 100 == 0:
            print(count, " DONE")
        matplotlib.image.imsave(save_path, image)


def image_processing(args):
    image_path = args.image_path
    models_dir = '/home/radice/neuralNetworks/robotcar-dataset-sdk/models'
    image_dir = image_path
    # Image Demosaicing and Undistortion
    # se il percorso è un file
    if os.path.isfile(image_path):
        camera_model = CameraModel(models_dir, image_dir)
        image = load_image(image_path, camera_model)
    # se il percorso è una cartella
    if os.path.isdir(image_path):
        sub = os.listdir(image_path)
        for dir in sub:
            if dir == 'left':
                process(models_dir, image_path, dir)
            if dir == 'right':
                process(models_dir, image_path, dir)


if __name__ == "__main__":
    args = parse_args()
    image_processing(args)