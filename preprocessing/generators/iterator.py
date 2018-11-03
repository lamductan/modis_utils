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

from keras.utils import np_utils
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img

import h5py


class SimpleIterator(Iterator):

    def __init__(self, batch_size, balancing=0, labels=None, n=None, shuffle=True, seed=None):

        if balancing == 0:

            assert (n is not None) or (labels is not None), \
                'Without rebalancing, either labels or n must be fed'
            if labels is not None:
                n = labels.shape[0]

            self.balance_threshold = balancing
            super().__init__(n, batch_size, shuffle, seed)
        else:
            num_data = labels.shape[0]
            super().__init__(num_data, batch_size, shuffle, seed)
            self.balance_threshold = balancing
            self.unique_labels = np.unique(labels)
            self.num_labels = len(self.unique_labels)

            self.inst_ids = {}

            labels = np.array(labels, dtype=np.int32)
            for lbl in self.unique_labels:
                self.inst_ids[int(lbl)] = np.flatnonzero(labels == int(lbl))

            if self.balance_threshold == -1:
                self.balance_threshold = max(
                    [len(self.inst_ids[i]) for i in self.inst_ids.keys() if i >= 0])

            self._set_index_array()

    def _set_index_array(self):

        if self.balance_threshold == 0:
            self.index_array = np.arange(self.n)
        else:
            self.index_array = np.zeros(
                (self.balance_threshold * self.num_labels,), 
                dtype=np.int32)

            for lbl in self.unique_labels:
                if lbl == -1: continue
                sub_labels = self.inst_ids[lbl]
                delta = self.balance_threshold - len(sub_labels)

                offset = lbl * self.balance_threshold
                if delta > 0:
                    self.index_array[offset: offset + self.balance_threshold] = \
                        np.r_[sub_labels, np.random.choice(sub_labels, delta)]
                elif delta < 0:
                    self.index_array[offset: offset + self.balance_threshold] = \
                        np.random.choice(sub_labels, self.balance_threshold)
                else:
                    self.index_array[offset: offset + self.balance_threshold] = \
                        sub_labels

            # NOTE Background class if presents
            if -1 in self.unique_labels:
                sub_labels = self.inst_ids[-1]
                self.index_array = np.r_[self.index_array, sub_labels]

            self.n = len(self.index_array)

        self.index_array = self.index_array
        if self.shuffle:
            np.random.shuffle(self.index_array)

    def __len__(self):

        return (self.n + self.batch_size - 1) // self.batch_size


