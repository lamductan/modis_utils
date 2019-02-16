import numpy as np
import os
import rasterio as rio
from scipy.ndimage import measurements
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.contrib.image import connected_components

from modis_utils.misc import get_buffer

CLOUD_FLAG = 0
WATER_FLAG = 1
LAND_WET_FLAG = 2
LAND_DRY_FLAG = 3
LABELS = [WATER_FLAG, LAND_WET_FLAG, LAND_DRY_FLAG]

WATER_THRESHOLD = {'NDVI': 1000}

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


def mask_lake_img(img, band='NDVI', offset=1000):
    offset = WATER_THRESHOLD[band]
    water_mask = np.where(img < offset, 1, 0)
    visited, label = measurements.label(water_mask)
    area = measurements.sum(water_mask, visited,
                            index = np.arange(label + 1))
    largest_element = np.argmax(area)
    return np.where(visited==largest_element, 1, 0)
    #return water_mask


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

def kmeans_mask(img, reservoir_index, quality=None):
    buffer = get_buffer(reservoir_index)
    pos = np.where(buffer==1)
    list_pixels = []
    for x, y in zip(*pos):
        list_pixels.append([img[x][y]])
    list_pixels = np.asarray(list_pixels)
    kmeans = KMeans(n_clusters=len(LABELS), random_state=0).fit(list_pixels)
    idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
    lut = np.zeros_like(idx)
    lut[idx] = LABELS 
    label = lut[kmeans.labels_]
    mask = np.zeros_like(img)
    i = 0
    for x, y in zip(*pos):
        mask[x][y] = label[i]
        i += 1

    if quality is not None:
        mask[np.where(quality > 0)] = CLOUD_FLAG
    return mask

    """
    def mask_water(img, offset):
        return np.where((offset[0]<=img) & (img<=offset[1]), 1, 0)

    def mask_out_lake(watermask):
        visited, label = measurements.label(watermask)
        area = measurements.sum(watermask, visited,
                index = np.arange(label + 1))
        largest_element = np.argmax(area)
        return np.where(visited==largest_element, 1, 0)

    def create_mask_lake(
            img,
            offset=MaskCloudWater.WATER_THRESHOLD):
        water_mask = mask_water(img, offset)
        return mask_out_lake(water_mask)

    def create_mask_lake_by_dir(
            img_dir,
            offset=MaskCloudWater.WATER_THRESHOLD):
        ds = rio.open(img_dir)
        band = ds.read(1)
        return create_mask_lake(band, offset)

    # Cloud
    def mask_cloud(img, offset):
        return np.where((offset[0]<=img) & (img<=offset[1]), 1.0, 0.0)

    def mask_out_cloud_in_lake(cloudmask, cloud_flag=0.5):
        visited, label = measurements.label(cloudmask)
        area = measurements.sum(cloudmask, visited,
                index = np.arange(label + 1))
        largest_element = np.argmax(area)
        return np.where(visited==largest_element, cloud_flag, 0.0)

    def create_mask_cloud_in_lake(
            img,
            offset=MaskCloudWater.CLOUD_THRESHOLD):
        cloud_mask = mask_cloud(img, offset)
        return mask_out_cloud_in_lake(cloud_mask)

    def create_mask_cloud_in_lake_by_dir(
            imgDir,
            offset=MaskCloudWater.CLOUD_THRESHOLD):
        ds = rio.open(imgDir)
        band = ds.read(1)
        return create_mask_cloud_in_lake(band, offset)

    # Cloud and water mask
    def create_cloud_and_water_mask_lake(
            img,
            water_offset=MaskCloudWater.WATER_THRESHOLD,
            cloud_offset=MaskCloudWater.CLOUD_THRESHOLD):
        cloud_mask = create_mask_cloud_in_lake(img, cloud_offset)
        water_mask = create_mask_lake(img, water_offset)
        cloud_and_water_mask = cloud_mask + waterMask
        cloud_and_water_mask[cloud_and_water_mask > 1.0] = 1.0
        return cloud_and_water_mask
    """

