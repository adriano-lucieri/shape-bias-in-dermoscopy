from PIL import Image

import numpy as np
import cv2
import os


def make_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_array(array):
    return (array/np.max(array))*255

def save_mask(mask_array, filepath, preprocessing='Normalize'):

    mask_array = np.squeeze(mask_array)

    if preprocessing == 'Normalize':
        mask_array = normalize_array(mask_array)

    img = Image.fromarray(mask_array.astype(np.uint8))
    img.save(filepath)

def plot_overlays(maps, filenames, destpath='/home/lucieri/Tmp/', im_size=299, im_channels=3,
                  range='pos', invert=False, color_map=cv2.COLORMAP_JET):
    """Plots canvas with original image, overlay and saliency map. (3 images of im_size width and height)

    Args:
        maps: List of saliency maps to plot []
        filenames: List of absolute filepaths to original images
        destpath: Absolute destination paths in which to save results.
        im_size: Single output image size.
        im_channels: Image channels
    """

    eps = 10e-7

    print(len(maps))
    print(maps[0].shape)

    # Switch dimensions from channels first to channels last
    maps = [np.squeeze(m) for m in maps]

    make_directory_if_not_exists(destpath)

    filepaths = []

    for idx, (map, src_filepath) in enumerate(zip(maps, filenames)):

        canvas = np.zeros((im_size, im_size*3, im_channels))

        if range == 'pos':

            if invert:
                map = 1.0 - map

            if len(map.shape) == 2:
                map = np.expand_dims(map, axis=-1)
                map = np.repeat(map, im_channels, axis=-1)
            map = cv2.resize(map, (im_size, im_size))

            heatmap = cv2.applyColorMap(np.uint8(255 * map), color_map)

        elif range == 'both':

            map = np.where(map > 0, map/(eps + np.max(map)), -(map/(-eps+np.min(map))))  # scale to [-1.0, 1.0]
            if len(map.shape) == 2:
                map = np.expand_dims(map, axis=-1)
                map = np.repeat(map, im_channels, axis=-1)
            map = cv2.resize(map, (im_size, im_size))

            heatmap = cv2.applyColorMap(np.uint8(128 * map + 128), color_map)

        img = cv2.imread(src_filepath)
        img = cv2.resize(img, (im_size, im_size))

        overlay = cv2.addWeighted(img, 1, heatmap, 0.2, 0)

        canvas[:, :im_size] = img
        canvas[:, im_size:im_size*2] = overlay
        canvas[:, 2*im_size:] = heatmap

        filename = src_filepath.split('/')[-1]
        filename = filename.split('.')[0]
        filename = ".".join([filename, 'png'])
        filepath = os.path.join(destpath, filename)
        filepaths.append(filepath)

        cv2.imwrite(filepath, canvas)
