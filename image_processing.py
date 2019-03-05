import os
import numpy as np
import rasterio as rio
import tensorflow as tf
from scipy.ndimage import measurements
from tensorflow.contrib.image import connected_components
from modis_utils.misc import restore_data, cache_data, get_im

CLOUD_FLAG = 0
WATER_FLAG = 1
LAND_WET_FLAG = 2
LAND_DRY_FLAG = 3
LABELS = [WATER_FLAG, LAND_WET_FLAG, LAND_DRY_FLAG]


def create_water_cloud_mask(data_dir, used_band, year_range,
                            n_data_per_year, day_period, mask_data_dir):
    for year in range(year_range[0], year_range[1] + 1):
        for d in range(n_data_per_year):
            day = d*day_period + 1
            prefix = os.path.join(str(year), str(year) + str(day).zfill(3))
            current_data_dir = os.path.join(data_dir, prefix)
            try:
                water_cloud_mask = mask_cloud_and_water(current_data_dir, used_band)
                cur_mask_data_dir = os.path.join(mask_data_dir, prefix)
                if not os.path.exists(cur_mask_data_dir):
                    os.makedirs(cur_mask_data_dir)
                cache_data(
                    water_cloud_mask, os.path.join(cur_mask_data_dir, 'masked.dat'))
            except:
                print('Not found band {} in {}{:03} in {}.'.format(
                    used_band, year, day, current_data_dir))


def create_groundtruth_mask_lake(data_dir, used_band, year_range, n_data_per_year,
                                 day_period, groundtruth_mask_lake_dir):
    for year in range(year_range[0], year_range[1] + 1):
        for d in range(n_data_per_year):
            day = d*day_period + 1
            prefix = os.path.join(str(year), str(year) + str(day).zfill(3))
            current_data_dir = os.path.join(data_dir, prefix)
            try:
                list_imgs = os.listdir(current_data_dir)
                band_filename = list(filter(lambda x: used_band in x, list_imgs))[0]
                img = rio.open(os.path.join(current_data_dir, band_filename), 'r').read(1)
                groundtruth_mask_lake = mask_lake_img(img, offset=1000)
                cur_mask_data_dir = os.path.join(groundtruth_mask_lake_dir, prefix)
                if not os.path.exists(cur_mask_data_dir):
                    os.makedirs(cur_mask_data_dir)
                cache_data(
                    groundtruth_mask_lake, os.path.join(cur_mask_data_dir, 'masked.dat'))
            except:
                print('Not found band {} in {}{:03} in {}.'.format(
                    used_band, year, day, current_data_dir))


def change_fill_value(data_dir, used_band, year_range, n_data_per_year,
                      day_period, change_fill_value_data_dir):
    for year in range(year_range[0], year_range[1] + 1):
        for d in range(n_data_per_year):
            day = d*day_period + 1
            prefix = os.path.join(str(year), str(year) + str(day).zfill(3))
            current_data_dir = os.path.join(data_dir, prefix)
            try:
                list_imgs = os.listdir(current_data_dir)
                band_filename = list(filter(lambda x: used_band in x, list_imgs))[0]
                img = get_im(os.path.join(current_data_dir, band_filename))
                img[img == -3000] = -2001
                cur_dest_dir = os.path.join(
                    change_fill_value_data_dir, prefix)
                if not os.path.exists(cur_dest_dir):
                    os.makedirs(cur_dest_dir)
                cache_data(img, os.path.join(cur_dest_dir, 'change_fill_value.dat'))
            except:
                print('Not found band {} in {}{:03} in {}.'.format(
                    used_band, year, day, current_data_dir))


def mask_cloud_and_water(img_dir, band='NDVI', offset=1000, cloud_flag=True):
    list_imgs = os.listdir(img_dir)
    band_filename = list(filter(lambda x: band in x, list_imgs))[0]
    band_filename = os.path.join(img_dir, band_filename)
    quality_filename = list(filter(lambda x: 'Quality' in x, list_imgs))[0]
    quality_filename = os.path.join(img_dir, quality_filename)
    with rio.open(band_filename, 'r') as img_src, \
         rio.open(quality_filename, 'r') as quality_src:
        img = img_src.read(1)
        quality = quality_src.read(1)
        quality1 = np.mod(quality, 4)
       
        mask = np.ones_like(img)*-1
        mask[np.where(img < offset)] = WATER_FLAG
        mask[np.where(quality1 >= 3)] *= 2
        mask[mask == -2] = CLOUD_FLAG
        mask[mask == 2] = WATER_FLAG
        return mask 


def mask_water_tf(img_tf, offset=0.1):
    return tf.where(tf.less(img_tf, offset),
                    tf.fill(tf.shape(img_tf), 1.0),
                    tf.fill(tf.shape(img_tf), -1.0))


def mask_lake_img(img, band='NDVI', offset=1000):
    water_mask = np.where(img < offset, 1, 0)
    visited, label = measurements.label(water_mask)
    area = measurements.sum(water_mask, visited,
                            index = np.arange(label + 1))
    largest_element = np.argmax(area)
    return np.where(visited==largest_element, 1, 0)


def mask_lake_img_tf(img_tf, band='NDVI'):
    offset = WATER_THRESHOLD[band]/10000
    water_mask_tf = tf.where(tf.less(img_tf, offset),
                             tf.fill(tf.shape(img_tf), 1),
                             tf.fill(tf.shape(img_tf), 0))
    def f1():
        visited_tf = connected_components(water_mask_tf)
        visited_tf_flatten = tf.reshape(visited_tf, [-1])
        mask = tf.not_equal(visited_tf_flatten, 0)
        non_zero_array = tf.boolean_mask(visited_tf_flatten, mask)
        y, _, area = tf.unique_with_counts(non_zero_array)    
        pos = tf.argmax(area)
        largest_element = tf.to_int32(y[pos])
        return tf.to_float(tf.where(tf.equal(visited_tf, largest_element),
                                    tf.fill(tf.shape(img_tf), 1.0),
                                    tf.fill(tf.shape(img_tf), 0.0)))
    def f2():
        return tf.to_float(water_mask_tf)
    return tf.cond(tf.greater(tf.reduce_max(water_mask_tf), 0), f1, f2)