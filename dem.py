import os
import numpy as np
import numpy.ma as ma
import gdal
import rasterio as rio
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

def find_rect_bound(polygon):
    xmin, ymin, xmax, ymax = polygon.bounds
    w, h = xmax - xmin, ymax - ymin
    xmin -= 0.1*w
    xmax += 0.1*w
    ymin -= 0.1*h
    ymax += 0.1*h
    return xmin, ymin, xmax, ymax

def get_dem(reservoir_index):
    dem_path = os.path.join('DEM', '{}.tif'.format(reservoir_index))
    with rio.open(dem_path, 'r') as src:
        return src.read(1)

def get_water_level(reservoir_index, mask_lake):
    dem = get_dem(reservoir_index)
    mask_lake_reversed = np.logical_xor(np.ones_like(mask_lake), mask_lake)
    mask_lake_dem = ma.masked_array(dem, mask_lake_reversed)
    return mask_lake_dem.max() - mask_lake_dem.min()

