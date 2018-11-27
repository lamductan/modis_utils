import gdal
import rasterio
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

def find_rect_bound(polygon):
    xmin, ymin, xmax, ymax = polygon.bounds
    w, h = xmax - xmin, ymax - ymin
    xmin -= 0.5*w
    xmax += 0.5*w
    ymin -= 0.5*h
    ymax += 0.5*h
    return xmin, ymin, xmax, ymax


