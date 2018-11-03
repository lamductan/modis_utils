#!/usr/bin/env python3
import os
import sys
import cv2
import logging
import argparse
import numpy as np

from .fast_transforms import *
from .image_iterator import ImageIterator

from keras.utils import np_utils
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img

import h5py

class ImageSequenceIterator(ImageIterator):

    def __init__(self, 
        x, 
        y, 
        image_generator, 
        sample_weights=None, 
        nframes=1, 
        reuse_perturbation=True,
        batch_size=32, 
        shuffle=False, 
        seed=None, 
        is_training=True,
        balancing=0):

        ImageIterator.__init__(self, 
            x, 
            y, 
            image_generator, 
            sample_weights=sample_weights, 
            batch_size=batch_size,
            shuffle=shuffle, 
            seed=seed, 
            is_training=is_training,
            balancing=balancing)

        self.nframes = nframes
        self.reuse_perturbation = reuse_perturbation

    def _get_batches_of_transformed_samples(self, index_array):

        # if self.channels_axis == 3:
        #     img_shape = tuple(self.image_jittering_generator.crop_size) + (3,)
        # else:
        #     img_shape = (3,) + np.zeros(tuple(self.image_jittering_generator.crop_size))
        img_shape = tuple(self.image_jittering_generator.crop_size)
        batch_x = np.zeros(tuple([len(index_array)]) + (self.nframes,) + img_shape, dtype=K.floatx())

        w = self.x.shape[3]
        h = self.x.shape[2]
        for i, j in enumerate(index_array):

            assert len(self.x[j]) >= 1, 'Image sequence at index %j must be non-empty' % j

            batch = []

            if self.reuse_perturbation:
                T = self.image_jittering_generator.random_transform()
                T['offset_x'] = np.random.randint(T['crop_size'][1] - w//2, w//2 - T['crop_size'][1])
                T['offset_y'] = np.random.randint(T['crop_size'][0] - h//2, h//2 - T['crop_size'][0])

            assert len(self.x[j]) == self.nframes

            for k, img in enumerate(self.x[j]):

                x = self.x[j][k].astype(K.floatx())

                if self.is_training:
                    if not self.reuse_perturbation:
                        T = self.image_jittering_generator.random_transform()
                        T['offset_x'] = np.random.randint(T['crop_size'][1]//2 - w//2, w//2 - T['crop_size'][1]//2)
                        T['offset_y'] = np.random.randint(T['crop_size'][0]//2 - h//2, h//2 - T['crop_size'][0]//2)
                    x = transform_image(x, T)
                x = self.image_jittering_generator.standardize(x.astype(K.floatx()))
                batch.append(x.astype(K.floatx())[np.newaxis, ...])

            batch_x[i] = np.concatenate(batch, axis=0)

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

        #batch_y = self.y[index_array]

        if self.sample_weights is not None:
            batch_weights = self.sample_weights[index_array]
            return batch_x, batch_y, batch_weights

        return batch_x, batch_y
