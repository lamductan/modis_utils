import numpy as np
from scipy import misc
import cv2
import csv
import pickle
import os
import rasterio as rio
#import tensorflow as tf

#List of years for MOD13Q1 only
TRAIN_LIST_YEARS_DEFAULT = [2000, 2001, 2002, 2003, 2004, 
                            2005, 2006, 2007, 2008, 2009,
                            2010, 2011]
VAL_LIST_YEARS_DEFAULT = [2012, 2013, 2014]
TEST_LIST_YEARS_DEFAULT = [2015, 2016, 2017]

'''
# List of year for combined MOD13Q1 and MYD13Q1
TRAIN_LIST_YEARS_DEFAULT = [2003, 2004, 2005, 2006, 2007,
                            2008, 2009, 2010, 2011]
VAL_LIST_YEARS_DEFAULT = [2012, 2013, 2014]
TEST_LIST_YEARS_DEFAULT = [2015, 2016, 2017]
'''

def to_str(x):
    if not isinstance(x, list):
        return str(x)
    else:
        b = [str(y) for y in x]
        return '_'.join(b)


def get_list_years_default(data_type):
    if data_type == 'train':
        return TRAIN_LIST_YEARS_DEFAULT
    elif data_type == 'val':
        return VAL_LIST_YEARS_DEFAULT
    else:
        return TEST_LIST_YEARS_DEFAULT


def get_dir_prefix(used_reservoir, used_band, time_steps):
    return os.path.join(str(time_steps), str(used_reservoir), used_band)


def get_data_file_dir(data_dir, used_band, input_time_steps, output_time_steps):
    return os.path.join('data_file', data_dir, str(input_time_steps), 
                        str(output_time_steps))


def get_data_file_path(data_dir, used_band, input_time_steps,
                       output_time_steps, data_type, file_type):
    return os.path.join(get_data_file_dir(data_dir, used_band, 
                                          input_time_steps, output_time_steps),
                        '{}_{}.csv'.format(data_type, file_type))


def get_cache_dir(data_dir, used_reservoir, used_band, time_steps):
    return os.path.join('cache', data_dir,
                        get_dir_prefix(used_reservoir, used_band, time_steps))


def get_cache_file_path(data_dir, used_reservoir, used_band,
                        time_steps, data_type):
    return os.path.join(get_cache_dir(data_dir, used_reservoir,
                                      used_band, time_steps),
                        '{}.dat'.format(data_type))


def get_data_augment_dir(data_dir, used_reservoir, used_band,
                         time_steps, data_type='', crop_size=32):
    return os.path.join('data_augment', data_dir, str(crop_size),
                        get_dir_prefix(used_reservoir, used_band, time_steps),
                        data_type)


def get_result_dir_suffix(data_dir, used_reservoir, used_band, crop_size,
                          time_steps, filters, kernel_size, n_hidden_layers,
                          mask_cloud_loss):
    return os.path.join(data_dir, str(crop_size), str(time_steps),
			used_band, to_str(filters), to_str(kernel_size),
			str(n_hidden_layers), str(mask_cloud_loss),
                        str(used_reservoir))


def get_result_dir(data_dir, used_reservoir, used_band, crop_size,
                   time_steps, filters, kernel_size, n_hidden_layers,
                   mask_cloud_loss):
    return os.path.join('result', get_result_dir_suffix(
        data_dir, used_reservoir, used_band, crop_size, time_steps,
        filters, kernel_size, n_hidden_layers, mask_cloud_loss))
 

def get_predict_dir(data_dir, used_reservoir, used_band, crop_size,
                    time_steps, filters, kernel_size, n_hidden_layers,
                    mask_cloud_loss):
    return os.path.join('predict', get_result_dir_suffix(
        data_dir, used_reservoir, used_band, crop_size, time_steps,
        filters, kernel_size, n_hidden_layers, mask_cloud_loss))
 

def get_predict_mask_dir(data_dir, used_reservoir, used_band, crop_size,
                        time_steps, filters, kernel_size, n_hidden_layers,
                        mask_cloud_loss):
    return os.path.join('predict_mask', get_result_dir_suffix(
        data_dir, used_reservoir, used_band, crop_size, time_steps,
        filters, kernel_size, n_hidden_layers, mask_cloud_loss))
 

def get_data_augment_merged_dir(data_dir, used_band, time_steps,
                                data_type='', crop_size=32):
    return os.path.join('data_augment_merged', data_dir, str(crop_size),
                        str(time_steps), used_band, data_type)

def get_threshold_mask_dir(modis_product, reservoir_index):
    return os.path.join('threshold_mask', modis_product, str(reservoir_index))

def get_threshold_mask_path(modis_product, reservoir_index, year, day):
    return os.path.join(get_threshold_mask_dir(
        modis_product, reservoir_index), '{}{:03}.dat'.format(year, day))
    
def get_threshold_mask(modis_product, reservoir_index, year, day):
    path = get_threshold_mask_path(modis_product, reservoir_index, year, day)
    if os.path.isfile(path):
        return restore_data(path)
    else:
        return None

def get_percentile_path(reservoir_index):
    return os.path.join('percentile', '{}.dat'.format(reservoir_index))

def get_percentile(reservoir_index):
    return restore_data(get_percentile_path(reservoir_index))

def get_buffer_path(reservoir_index):
    return os.path.join('buffer', '{}.dat'.format(reservoir_index))

def get_buffer(reservoir_index):
    return restore_data(get_buffer_path(reservoir_index))

def get_kmeans_mask_path(data_dir, reservoir_index, year, day):
    return os.path.join('kmeans_mask', data_dir, str(reservoir_index),
                        '{}{:03}.dat'.format(year, day))

def get_mask_zone(reservoir_index):
    return restore_data(os.path.join('mask_zone', '{}.dat'.format(reservoir_index)))

def get_quality_img(modis_product, reservoir_index, year, day):
    img_dir = os.path.join('raw_data', modis_product, str(reservoir_index),
                           str(year), '{}{:03}'.format(year, day))
    list_imgs = os.listdir(img_dir)
    quality_filename = list(filter(lambda x: 'reliability' in x, list_imgs))[0]
    quality_filename = os.path.join(img_dir, quality_filename)
    return restore_data(quality_filename)

def get_mask_dir(modis_product, reservoir_index, year, day, Zhang=False):
    if Zhang:
        return os.path.join('Zhang_mask_data', modis_product, str(reservoir_index),
                        str(year), '{}{:03}'.format(year, day))
    return os.path.join('mask_data', modis_product, str(reservoir_index),
                        str(year), '{}{:03}'.format(year, day))

def get_mask_path(modis_product, reservoir_index, year, day, Zhang=False):
    return os.path.join(get_mask_dir(modis_product, reservoir_index,
                                     year, day, Zhang),
                        'masked.dat')

def get_mask(modis_product, reservoir_index, year, day, Zhang=False):
    try:
        if Zhang:
            path = get_Zhang_mask_path(modis_product, reservoir_index, 
                                       year, day)
        else:
            path = get_mask_path(modis_product, reservoir_index, year, day)
        return restore_data(path)
    except:
        return None


# Data
def cache_data(data, path):
    """Save data (numpy array) to disk."""
    token = path.split('/')
    dir_prefix = ''
    for t in token[:-1]:
        dir_prefix = os.path.join(dir_prefix, t)
    try:
        os.makedirs(dir_prefix)
    except:
        pass
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def to_float32(x):
    if not isinstance(x, tuple):
        return np.float32(x)
    b = []
    for a in x:
        b.append(to_float32(a))
    return tuple(b)


def restore_data(path, convert_to_float32=False):
    """Restore cached data from disk to memory."""
    if path[-4:] != '.dat':
        return get_im(path)

    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
        if convert_to_float32:
            data = to_float32(data)
    return data


def _get_data_path(data_file):
    with open(data_file, "r") as data_f:
        reader = csv.reader(data_f)
        data_paths = [row for row in reader]
    return data_paths


def get_data_paths(data_file):
    return _get_data_path(data_file)


def _get_target_mask_path(target_file):
    with open(target_file, "r") as target_f:
        reader = csv.reader(target_f)
        target_paths = [row for row in reader]
    return target_paths


def get_target_paths(target_file):
    return _get_target_mask_path(target_file)


def get_mask_paths(mask_file):
    return _get_target_mask_path(mask_file)


def get_im(path, reduce_size=None):
    # reduce_size type is a tuple, example: (128, 96)
    img = misc.imread(path)
    # Reduce size
    if reduce_size is not None:
        img = misc.imresize(img, reduce_size)
    return img


def _load_data(data_file, reduce_size=None):
    data_paths = _get_data_path(data_file)
    X_ = []
    for data_path_list in data_paths:
        currentX_ = []
        for fl in data_path_list:
            img = restore_data(fl)
            currentX_.append(img)
        X_.append(currentX_)
    return X_


def _load_target_mask(target_file, reduce_size=None):
    target_mask_paths = _get_target_mask_path(target_file)
    y_ = []
    for path in target_mask_paths:
        if path[-4:] != '.dat':
            img = get_im(path, reduce_size)
        else:
            img = restore_data(path)
        y_.append(img)
    return y_


def check_years_in_file(target_file, list_years):
    paths = _get_target_mask_path(target_file)
    list_years_in_file = list(map(int, [path.split('/')[-3] 
                                        for path in paths]))
    list_years_in_file = list(set(list_years_in_file))
    list_years_in_file.sort()
    list_years.sort()
    return list_years_in_file == list_years


def normalize_data(data, mean=0.0, std=1.0):
    if std == 0:
        return data - mean
    return (data - mean) / std

#def scale_data(data, min=-1000, max=10000, range_min=0, range_max=1):
#    return np.interp(data, (min, max), (range_min, range_max))

def scale_data(data, original_range=(-1.0,1.0), range=(-0.2001,1.0)):
    return np.interp(data, original_range, range)

def scale_normalized_data(normalized_data, range=(-1.0,1.0)):
    return np.interp(normalized_data, (-0.2001, 1.0), range)

def scale_data_tf(data_tf, original_range=(-1.0, 1.0), output_range=(-0.2001, 1.0)):
    original_diff = original_range[1] - original_range[0]
    output_diff = output_range[1] - output_range[0]
    data_zero_one_scale = tf.divide(tf.subtract(data_tf, original_range[0]), original_diff)
    return tf.add(tf.multiply(data_zero_one_scale, output_diff), output_range[0])

def scale_data_with_scaler(data, scaler):
    """Scale data to [0, 1].
    
    Args:
        data: Data need to be scaled, prefers numpy array.
        scaler: MinMaxScaler object
    
    Returns:
    Data after scaled.
    """
    if (data.ndim == 5):
        x = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], 
                         data.shape[3]*data.shape[4])
    else:
        x = data.reshape(data.shape[0]*data.shape[1], 
                         data.shape[2]*data.shape[3])
    scale_x = scaler.transform(x)
    return scale_x.reshape(data.shape)


def find_img_name(data_dir='raw_data/MOD13Q1',
                  reservoir_index=0, year=2001,
                  day=1, band_find='NDVI'):
    dir = os.path.join(data_dir, str(reservoir_index), str(year),
                       str(year) + str(day).zfill(3))
    try:
        list_raster = os.listdir(dir)
        find = list(filter(lambda x: band_find in x, list_raster))
        return dir + '/' + find[0]
    except:
        return None


def find_img_name_1(data_dir='raw_data/MOD13Q1',
                    reservoir_index=0, year=2001,
                    day=1, band_find='NDVI'):
    dir = os.path.join(data_dir, str(reservoir_index), str(year),
                       str(year) + str(day).zfill(3))
    try:
        list_raster = os.listdir(dir)
        find = list(filter(lambda x: band_find in x, list_raster))
        return find[0]
    except:
        return None


def _get_list_data_in_one_year(data_dir, used_band, year, mask=False):
    if mask:
        filename = 'masked.dat'
    else:
        filename = 'preprocessed.dat'
    year_dir = os.path.join(data_dir, str(year))
    list_folders = os.listdir(year_dir)
    list_folders = sorted(list_folders, key=lambda x: int(x))
    return [os.path.join(year_dir, x, filename) for x in list_folders]


def _create_data_file_continuous_years(data_dir,
                                       input_time_steps,
                                       output_time_steps,
                                       list_years,
                                       data_type,
                                       mask_data_dir,
                                       used_band=''):
    input_file = get_data_file_path(data_dir, used_band, input_time_steps,
                                    output_time_steps, data_type, 'data') 
    target_file = get_data_file_path(data_dir, used_band, input_time_steps,
                                     output_time_steps, data_type, 'target') 
    mask_file = get_data_file_path(data_dir, used_band, input_time_steps,
                                   output_time_steps, data_type, 'mask')
    input_f = open(input_file, 'w')
    target_f = open(target_file, 'w')
    mask_f = open(mask_file, 'w')
    writer_input = csv.writer(input_f)
    writer_target = csv.writer(target_f)
    writer_mask = csv.writer(mask_f)

    list_data = []
    list_mask = []

    if list_years[0] > 2000:
        list_data_prev_year = _get_list_data_in_one_year(
            data_dir, used_band, list_years[0] - 1)
        list_data += list_data_prev_year[-input_time_steps:]
        list_mask_prev_year = _get_list_data_in_one_year(
            mask_data_dir, used_band, list_years[0] - 1, True)
        list_mask += list_mask_prev_year[-input_time_steps:]

    for year in list_years:
        list_data += _get_list_data_in_one_year(data_dir, used_band, year)
        list_mask += _get_list_data_in_one_year(mask_data_dir, used_band, year, True)

    if list_years[-1] < 2018 and output_time_steps > 1:
        list_data_next_year = _get_list_data_in_one_year(
            data_dir, used_band, list_years[-1] + 1)
        list_data += list_data_next_year[:output_time_steps - 1]
        list_mask_next_year = _get_list_data_in_one_year(
            mask_data_dir, used_band, list_years[-1] + 1, True)
        list_mask += list_mask_next_year[:output_time_steps - 1]

    n_data = len(list_data) - (input_time_steps + output_time_steps) + 1
    for i in range(input_time_steps, n_data + input_time_steps):
        list_data_in_window = list_data[i - input_time_steps : i]
        list_target_in_window = list_data[i : i + output_time_steps]
        list_mask_in_window = list_mask[i : i + output_time_steps]
        writer_input.writerow(list_data_in_window)
        writer_target.writerow(list_target_in_window)
        writer_mask.writerow(list_mask_in_window)

    input_f.close()
    target_f.close()
    mask_f.close()


def create_data_file_continuous_years(data_dir='raw_data/MOD13Q1',
                                      input_time_steps=12,
                                      output_time_steps=1,
                                      list_years_train=None,
                                      list_years_val=None,
                                      list_years_test=None,
                                      mask_data_dir='masked_data/MOD13Q1',
                                      used_band=''):
    """Create files containing path of images.
    
    If you already created those files but now you change list_years,
    you must remove all old files and recreate them.
    
    Example:
        create_data_file_continuous_years(data_dir='raw_data/MOD13Q1/0',
                                          used_band='NDVI',
                                          input_time_steps=12,
                                          output_time_steps=1,
                                          list_years_train=None,
                                          list_years_val=None,
                                          list_years_test=None,
                                          mask_data_dir='masked_data/MOD13Q1'):

    Args:
        data_dir: Directory where stores image data.
        input_time_steps: Input Time steps (length) of LSTM sequence.
        output_time_steps: Output Time steps (length) of LSTM sequence.
        list_years_train: List years of data used for train, use None if
            want to use default range.
        list_years_val: List years of data used for validation.
        list_years_test: List years of data used for test.
        mask_data_dir: Directory where stores masked images.
        used_band: A string represents name of used band.

    Returns:
        A dictionary stores paths of data files.
    """
    list_years = {}
    data_types = ['train', 'val', 'test']
    file_types = ['data', 'target']
    if mask_data_dir is not None:
        file_types.append('mask')
        
    if list_years_train is None:
        list_years['train'] = TRAIN_LIST_YEARS_DEFAULT
    else:
        list_years['train'] = list_years_train
    
    if list_years_val is None:
        list_years['val'] = VAL_LIST_YEARS_DEFAULT
    else:
        list_years['val'] = list_years_val

    if list_years_test is None:
        list_years['test'] = TEST_LIST_YEARS_DEFAULT
    else:
        list_years['test'] = list_years_test

    data_file_dir = get_data_file_dir(data_dir, used_band,
                                      input_time_steps, output_time_steps)
    
    outputs = {}
    for data_type in data_types:
        outputs[data_type] = {}
        for file_type in file_types:
            outputs[data_type][file_type] = get_data_file_path(
                data_dir, used_band, input_time_steps,
                output_time_steps, data_type, file_type)
    
    # Check whether all needed files are created
    if os.path.isdir(data_file_dir):
        if mask_data_dir is None:
            return outputs
        elif os.path.isfile(outputs['train']['mask']):
            return outputs
    
    # Not all files are created, Create/Recreate them.
    try:
        os.makedirs(data_file_dir)
    except:
        pass
    for data_type in data_types:
        _create_data_file_continuous_years(data_dir, input_time_steps,
                                           output_time_steps, list_years[data_type],
                                           data_type, mask_data_dir, '')
    return outputs


def get_data_merged_from_paths(data_paths, target_paths, mask_paths):
    list_data = []
    for data_path in data_paths:
        list_data.append(np.expand_dims(restore_data(data_path), axis=0))
    for target_path in target_paths:
        list_data.append(np.expand_dims(restore_data(target_path), axis=0))
    for mask_path in mask_paths:
        list_data.append(np.expand_dims(restore_data(mask_path), axis=0))
    data_merged = np.concatenate(list_data, axis=0)
    data_merged = np.expand_dims(data_merged, axis=0)
    return data_merged


def get_target_test(target_file_path, which):
    target_paths_list = _get_target_mask_path(target_file_path)
    target_paths = target_paths_list[which]
    if len(target_paths) == 1:
        return restore_data(target_paths[0])
    else:
        list_target = []
        for path in target_paths:
            list_target.append(np.expand_dims(restore_data(path), axis=0))
        return np.concatenate(list_target, axis=0)


def get_data_test(data_file_path, which):
    data_paths_list = _get_data_path(data_file_path)
    data_paths = data_paths_list[which]
    list_data = []
    for path in data_paths:
        list_data.append(np.expand_dims(restore_data(path), axis=0))
    return np.concatenate(list_data, axis=0)


def get_reservoir_min_max(data_dir, reservoir_index):
    min_max_path = os.path.join('min_max', data_dir, 'min_max.dat')
    min_max = restore_data(min_max_path)
    reservoir_min = min_max[reservoir_index]['min']
    reservoir_max = min_max[reservoir_index]['max']
    return reservoir_min, reservoir_max


def get_reservoir_mean_std(data_dir, reservoir_index):
    data_dir = data_dir.strip('/')
    token = data_dir.split('/')
    preprocessed_dir = '/'.join(token[:2])
    modis_product = token[-1]
    mean_std_path = os.path.join('mean_std', preprocessed_dir, 'change_fill_value', modis_product, 
                                 'mean_std.dat')
    mean_std_dict = restore_data(mean_std_path)
    mean_std_dict_reservoir = mean_std_dict[reservoir_index]
    mean = mean_std_dict_reservoir['mean']
    std = mean_std_dict_reservoir['std']
    return mean, std

