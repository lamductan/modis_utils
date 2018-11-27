import numpy as np
import os
import rasterio as rio
from scipy.ndimage import measurements

WATER_THRESHOLD = {'NDVI': (-3000, 0)}

def mask_cloud_and_water(img_dir, band='NDVI'):
    offset = WATER_THRESHOLD[band]
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
        mask[np.where((offset[0]<=img) & (img<=offset[1]))] = 1
        mask[np.where(quality1 >= 2)] = 0
        return mask 


def mask_lake_img(img, band='NDVI'):
    offset = WATER_THRESHOLD[band]
    water_mask = np.where(((offset[0]<=img) & (img<=offset[1])), 1, 0)
    visited, label = measurements.label(water_mask)
    area = measurements.sum(water_mask, visited,
                            index = np.arange(label + 1))
    largest_element = np.argmax(area)
    return np.where(visited==largest_element, 1, 0)


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

