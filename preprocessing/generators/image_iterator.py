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

from .fast_transforms import *
from .iterator import SimpleIterator

from keras.utils import np_utils
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img

import h5py

class ImageIterator(SimpleIterator):

    def __init__(self, 
        x, y, 
        image_jittering_generator, 
        sample_weights=None,
        batch_size=32, 
        shuffle=False, 
        seed=None, 
        is_training=True,
        balancing=0,
        data_format=None, 
        save_to_dir=None, 
        save_prefix='', 
        save_format='png'):

        if y is not None and x.shape[0] != y.shape[0]:
            raise ValueError('X (images tensor) and y (labels) should have the same length.')

        if data_format is None: 
            data_format = K.image_data_format()

        self.x = x

        channels_axis = 3 if data_format == 'channels_last' else 1

        self.image_jittering_generator = image_jittering_generator
        self.channels_axis = channels_axis
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.is_training = is_training

        if y is not None:
            super().__init__(
                batch_size, 
                balancing=balancing, 
                labels=y, 
                shuffle=shuffle, 
                seed=seed)

        elif y is None and balancing == 0:
            self.y = None
            super().__init__(
                batch_size, 
                n=len(x), 
                shuffle=shuffle, 
                seed=seed)

        elif y is None and balancing != 0:
            raise ValueError('Label y must be fed if balancing is set')

        if y is not None:
            if np.max(y) > 1:
                nclasses = np.max(y) + 1
                self.y = np_utils.to_categorical(y, nclasses)

        self.sample_weights = sample_weights
        if self.sample_weights is not None:
            self.sample_weights = np.array(sample_weights, dtype=np.float32)

    def _get_batches_of_transformed_samples(self, index_array):

        if self.channels_axis == 3:
            img_shape = tuple(self.image_jittering_generator.crop_size) + (3,)
        else:
            img_shape = (3,) + tuple(self.image_jittering_generator.crop_size)

        batch_x = np.zeros(
            tuple([len(index_array)]) + img_shape, 
            dtype=K.floatx())

        for i, j in enumerate(index_array):

            x = self.x[j].astype(K.floatx())
            if self.is_training:
                T = self.image_jittering_generator.random_transform()
                x = transform_image(x, T)
            x = self.image_jittering_generator.standardize(x.astype(K.floatx()))

            batch_x[i] = x.astype(K.floatx())

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x

        batch_y = self.y[index_array]

        if self.sample_weights is not None:
            batch_weights = self.sample_weights[index_array]
            return batch_x, batch_y, batch_weights

        return batch_x, batch_y

    def __next__(self):

        with self.lock:
            index_array = next(self.index_generator)
        if isinstance(index_array, tuple):
            index_array = index_array[0]
        return self._get_batches_of_transformed_samples(index_array)

