# Transformation for image augmentation
# Author: Vo Dinh Phong
# Email: phong.vodinh@gmail.com
# Copyright by the same author
import os
import math
import time
import cv2
import logging
import argparse
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist

def img_zoom(img, fx, fy, interp=cv2.INTER_AREA):
    """
    Simulate zooming effect by upscaling/downscaling image using OpenCV
    """
    res = cv2.resize(img, None, fx=fx, fy=fy,
                     interpolation=interp)
    return res


def img_rotate(img, angle, center, fillval=0):
    """
    Rotate image with an angle, and rotation center.
    When the center is not fed, the default will be
    at the center of the image.
    """
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (cols, rows), borderValue=fillval)

def img_resize_crop(img, bbox, fx, fy, w, h, interp=cv2.INTER_AREA):
    ymin, xmin, ymax, xmax = bbox

    assert fx > 0 and fy > 0
    assert w > 0 and h > 0
    assert ymax > ymin and xmax > xmin, bbox

    # rscale the bbox wrt fx, fy, prior cropping them and really image resizing
    x = np.array([
        [xmin, ymin, 1.],
        [xmax, ymax, 1.]])

    cx, cy = (xmax + xmin) // 2, (ymax + ymin) // 2
    translate = np.array([
        [1., 0., -cx],
        [0., 1., -cy],
        [0., 0., 1.]])

    scale = np.array([
        [fx, 0., 0.],
        [0., fy, 0.],
        [0., 0., 1.]])

    xmin, ymin, _, xmax, ymax, _ = \
        np.abs(translate).dot(scale.dot(translate.dot(x.T))).T.ravel()

    img_h, img_w = img.shape[:2]

    xmin, xmax = int(max(xmin, 0)), int(min(xmax, img_w))
    ymin, ymax = int(max(ymin, 0)), int(min(ymax, img_h))

    if img.ndim == 3:
        cropped = img[ymin:ymax, xmin:xmax, :]
    elif img.ndim == 2:
        cropped = img[ymin:ymax, xmin:xmax]

    res = cv2.resize(
        cropped, (w, h),
        interpolation=interp)

    return res

def img_adjust_gamma_beta(img, gamma, beta):
    inv_gamma = 1. / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')
    img = cv2.LUT(img, table)
    return cv2.threshold(cv2.LUT(img, table) + beta, 256, 255, cv2.THRESH_TRUNC)[1]


def img_adjust_pixels(img, contrast, brightness):
    return cv2.convertScaleAbs(img, img, contrast, brightness)


def img_blurring(img, radius):
    return cv2.blur(img, (radius, radius))


def img_flip(img, axis=0):
    assert (axis == 0 or axis == 1)
    img = img.swapaxes(axis, 0)
    img = img[::-1, ...]
    img = img.swapaxes(0, axis)
    return img

def generate_random_transforms(
    crop_size, 
    random_crop=True, 
    auto_pre_zoom=False,
    rotation_range=(0, 0), 
    rotation_offset=(0, 0),
    zoom_range=(1, 1), 
    isotropic_zoom=False,
    contrast_range=(1, 1), 
    brightness_range=(0, 0),
    blurring_radius=0, 
    horizontal_flip=False, 
    vertical_flip=False,
    translate_range=(0, 0),
    fillval=0):

    transforms = {}

    transforms['crop_size'] = crop_size
    transforms['random_crop'] = random_crop
    transforms['angle'] = np.random.uniform(rotation_range[0], rotation_range[1])
    transforms['translate'] = np.random.uniform(translate_range[0], translate_range[1], 2)
    #transforms['offset'] = np.random.uniform(rotation_offset[0], rotation_offset[1], 2)
    transforms['fillval'] = fillval

    if zoom_range[0] != 1 or zoom_range[1] != 1:
        if isotropic_zoom:
            transforms['zy'] = np.random.uniform(zoom_range[0], zoom_range[1])
            transforms['zx'] = transforms['zy']
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
            transforms['zx'] = zx
            transforms['zy'] = zy
    else:
        transforms['zx'] = transforms['zy'] = 1.

    if contrast_range[0] <= contrast_range[1]:
        transforms['alpha'] = np.random.uniform(contrast_range[0], contrast_range[1])

    if brightness_range[0] <= brightness_range[1]:
        transforms['beta'] = np.random.uniform(brightness_range[0], brightness_range[1])

    if horizontal_flip:
        transforms['yflip'] = np.random.choice(2)

    if vertical_flip:
        transforms['xflip'] = np.random.choice(2)

    blurring_radius = np.random.choice([0, blurring_radius])
    transforms['blurring_radius'] = blurring_radius

    return transforms

def transform_image(image, T, constant_intensities=False, is_training=True):

    assert image.ndim >=2, 'Invalid ndimension %d' % image.ndim

    if not is_training:
        w, h = T['crop_size']
        cy, cx = image.shape[0]//2, image.shape[1]//2
        img = image[
            cy-h//2:cy+h//2, 
            cx-w//2:cx+w//2,
            :]
        return img

    if constant_intensities:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR

    cx, cy = image.shape[1] // 2, image.shape[0] // 2
    #cx += int(T['offset'][1] * cx * 2)
    #cy += int(T['offset'][0] * cy * 2)
    cx += T['offset_x']
    cy += T['offset_y']

    if not constant_intensities:
        img = img_rotate(image, T['angle'], (cx, cy), T['fillval']).astype(np.uint8)
    else:
        img = img_rotate(image, T['angle'], (cx, cy)).astype(np.uint8)

    box = cy - T['crop_size'][0]//2, cx - T['crop_size'][1]//2, \
        cy + T['crop_size'][0]//2, cx + T['crop_size'][1]//2

    height, width = image.shape[:2]
    delta_x = int(T['translate'][0]*width)
    delta_y = int(T['translate'][1]*height)
    box = [box[0]+delta_y, box[1]+delta_x, box[2]+delta_y, box[3]+delta_x]

    img = img_resize_crop(
        img,
        box,
        T['zx'], T['zy'],
        T['crop_size'][0], T['crop_size'][1],
        interp=interp)

    if not constant_intensities:
        if 'alpha' in T.keys() and 'beta' in T.keys():
            img = img_adjust_pixels(img, T['alpha'], T['beta'])
        if T['blurring_radius'] > 0:
            img = img_blurring(img, T['blurring_radius'])

    if 'xflip' in T.keys():
        img = img_flip(img, axis=1)

    if 'yflip' in T.keys():
        img = img_flip(img, axis=0)


    return img

