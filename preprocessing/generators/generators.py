#!/usr/bin/env python3

# Transformation for image augmentation
# Author: Vo Dinh Phong
# Email: phong.vodinh@gmail.com
# Copyright by the same author

import os
import sys
import cv2
import logging
import argparse
import numpy as np


from keras.utils import np_utils
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import array_to_img

from .fast_transforms import *
#from .masked_seq_iterator import MaskedImageSequenceIterator
from .image_iterator import ImageIterator
from .seq_iterator import ImageSequenceIterator

import h5py

class SimpleImageGenerator(ImageDataGenerator):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 crop_size=(299, 299),
                 random_crop=False,
                 rotation_range=(0, 0),
                 rotation_offset=(0, 0),
                 translate_range=(0, 0),
                 zoom_range=(1, 1),
                 isotropic_zoom=True,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 contrast_range=(1, 1),
                 brightness_range=(0, 0),
                 blurring_radius=0,
                 fillval=0,
                 preprocessing_function=None):

        ImageDataGenerator.__init__(
            self,
            featurewise_center,
            samplewise_center,
            featurewise_std_normalization,
            samplewise_std_normalization,
            zca_whitening,
            zca_epsilon,
            rotation_range,
            zoom_range=zoom_range,
            rescale=rescale,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            preprocessing_function=preprocessing_function)

        self.fillval = fillval
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.rotation_offset = rotation_offset
        self.isotropic_zoom = isotropic_zoom
        self.translate_range = translate_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.blurring_radius = blurring_radius

    def fit(self, images):

        self.mean = np.zeros((3,), dtype=float)
        self.std = np.zeros((3,), dtype=float)

        for image in images:
            mean = np.mean(image.astype(K.floatx()), axis=(0, 1))
            std = np.std(image, axis=(0, 1))
            if image.ndim == 2:
                mean = np.array([mean, mean, mean])
                std = np.array([std, std, std])
            self.mean += mean
            self.std += std

        self.mean /= len(images)
        self.std /= len(images)

        broadcast_shape = [1, 1, 1]
        broadcast_shape[self.channel_axis - 1] = 3
        self.mean = np.reshape(self.mean, broadcast_shape)
        self.std = np.reshape(self.std, broadcast_shape)

        logging.warning('ZCA Whitening not applicable')

    def flow(self):

        raise NotImplementedError('Given arbitrary image sizes, flow() is not implemented.')

    def flow_from_list(self, 
        x,
        mask=None,
        y=None,
        sample_weights=None,
        nframes=1,
        reuse_perturbation=True,
        batch_size=32,
        shuffle=True,
        seed=None,
        balancing=0,
        is_training=True,
        label_is_image=False,
        input_name='input',
        mask_name='mask_output',
        label_name='label_output'):

        if nframes > 1:

            if mask is None:
                return ImageSequenceIterator(
                    x, y, self,
                    sample_weights=sample_weights,
                    nframes=nframes,
                    reuse_perturbation=reuse_perturbation,
                    balancing=balancing,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    seed=seed,
                    is_training=is_training)
            else:
                return MaskedImageSequenceIterator(
                    x, mask, y, self,
                    nframes=nframes,
                    balancing=balancing,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    seed=seed,
                    label_is_image=label_is_image,
                    is_training=is_training,
                    input_name=input_name,
                    mask_name=mask_name,
                    label_name=label_name)
        else:

            return ImageIterator(
                x, y, self,
                sample_weights=sample_weights,
                balancing=balancing,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                is_training=is_training)

    def flow_from_directory(self, directory,
                            classes=None,
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png'):

        raise NotImplementedError('Implemented later')

    def random_transform(self, seed=None):

        if seed is not None:
            np.random.seed(seed)

        return generate_random_transforms(
            self.crop_size,
            fillval=self.fillval,
            random_crop=self.random_crop,
            rotation_range=self.rotation_range,
            rotation_offset=self.rotation_offset,
            translate_range=self.translate_range,
            zoom_range=self.zoom_range,
            isotropic_zoom=self.isotropic_zoom,
            contrast_range=self.contrast_range,
            brightness_range=self.brightness_range,
            blurring_radius=self.blurring_radius,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip)

