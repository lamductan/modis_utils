import numpy as np
from scipy import misc
import cv2
import csv
import pickle
import os

TRAIN_LIST_YEARS_DEFAULT = [2002, 2007, 2009, 2010, 2014, 
                            2005, 2003, 2015, 2011]
VAL_LIST_YEARS_DEFAULT = [2001, 2006, 2016, 2013]
TEST_LIST_YEARS_DEFAULT = [2008, 2017, 2012, 2004]

def get_list_years_default(data_type):
    if data_type == 'train':
        return TRAIN_LIST_YEARS_DEFAULT
    elif data_type == 'val':
        return VAL_LIST_YEARS_DEFAULT
    else:
        return TEST_LIST_YEARS_DEFAULT


def get_dir_prefix(used_reservoir, used_band, time_steps):
    return os.path.join(str(time_steps), str(used_reservoir), used_band)


def get_data_file_dir(data_dir, used_reservoir, used_band, time_steps):
    return os.path.join('data_file', data_dir,
                        get_dir_prefix(used_reservoir, used_band, time_steps))


def get_data_file_path(data_dir, used_reservoir, used_band,
                       time_steps, data_type, file_type):
    return os.path.join(get_data_file_dir(data_dir, used_reservoir,
                                          used_band, time_steps),
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


def get_data_augment_merged_dir(data_dir, used_band, time_steps,
                                data_type='', crop_size=32):
    return os.path.join('data_augment_merged', data_dir, str(crop_size),
                        str(time_steps), used_band, data_type)

    
def _create_data_file(data_dir,
                      used_reservoir,
                      used_band,
                      time_steps,
                      list_years,
                      data_type,
                      mask_data_dir):
    input_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                    time_steps, data_type, 'data') 
    target_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                     time_steps, data_type, 'target') 

    input_f = open(input_file, 'w')
    target_f = open(target_file, 'w')
    writer_input = csv.writer(input_f)
    writer_target = csv.writer(target_f)
    mask_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                   time_steps, data_type, 'mask')
    mask_f = open(mask_file, 'w')
    writer_mask = csv.writer(mask_f)

    time_steps += 1
    for year in list_years:
        list_files_in_window = []
        year_dir = os.path.join(data_dir, str(used_reservoir), str(year))
        if mask_data_dir is not None:
             mask_year_dir = os.path.join(mask_data_dir, 
                                          str(used_reservoir), str(year))
        list_folders = os.listdir(year_dir)
        list_folders = sorted(list_folders, key=lambda x: int(x))
        
        # Write 1st timesteps
        day = 0
        for i in np.arange(time_steps):
            day = list_folders[i]
            list_files = os.listdir(os.path.join(year_dir, day))
            for file in list_files:
                if used_band in file:
                    list_files_in_window.append(
                        os.path.join(day, file))
        writer_input.writerow([os.path.join(year_dir, file_path)
            for file_path in list_files_in_window[:-1]])
        writer_target.writerow([os.path.join(year_dir, file_path)
            for file_path in list_files_in_window[-1:]])
        writer_mask.writerow([os.path.join(mask_year_dir, day,
                              'masked.tif')])

        # write 2nd to last timesteps
        for i in np.arange(time_steps, len(list_folders)):
            day = list_folders[i]
            list_files_in_window = list_files_in_window[1:]
            list_files = os.listdir(os.path.join(year_dir, day))
            for file in list_files:
                if used_band in file:
                    list_files_in_window.append(
                        os.path.join(day, file))
            writer_input.writerow([os.path.join(year_dir, file_path)
                for file_path in list_files_in_window[:-1]])
            writer_target.writerow([os.path.join(year_dir, file_path)
                for file_path in list_files_in_window[-1:]])
            writer_mask.writerow([os.path.join(mask_year_dir, day,
                                  'masked.tif')])

    input_f.close()
    target_f.close()
    mask_f.close()


def create_data_file(data_dir='raw_data/MOD13Q1',
                     used_reservoir=0,
                     used_band='NDVI',
                     time_steps=12,
                     train_list_years=None,
                     val_list_years=None,
                     test_list_years=None,
                     mask_data_dir='mask_data/MOD13Q1'):
    """Create files containing path of images.
    
    If you already created those files but now you change list_years,
    you must remove all old files and recreate them.
    
    Example:
        create_data_file(data_dir='raw_data/MOD13Q1',
                         used_reservoir=0,
                         used_band='NDVI',
                         time_steps=12,
                         train_list_years=None,
                         val_list_years=None,
                         test_list_years=None,
                         mask_data_dir='mask_data/MOD13Q1'):

    Args:
        data_dir: Directory where stores image data.
        used_reservoir: Index of processed reservoir.
        used_band: A string represents name of used band.
        time_steps: Time steps (length) of LSTM sequence.
        train_list_years: List years of data used for train, use None if
            want to use default range.
        val_list_years: List years of data used for validation.
        test_list_years: List years of data used for test.
        mask_data_dir: Directory where stores masked images.

    Returns:
        A dictionary stores paths of data files.
    """
    list_years = {}
    data_types = ['train', 'val', 'test']
    file_types = ['data', 'target']
    if mask_data_dir is not None:
        file_types.append('mask')
        
    if train_list_years is None:
        list_years['train'] = [2002, 2007, 2009, 2010, 2014, 
                            2005, 2003, 2015, 2011]
    else:
        list_years['train'] = train_list_years
    
    if val_list_years is None:
        list_years['val'] = [2001, 2006, 2016, 2013]
    else:
        list_years['val'] = val_list_years

    if test_list_years is None:
        list_years['test'] = [2008, 2017, 2012, 2004]
    else:
        list_years['test'] = test_list_years

    data_file_dir = get_data_file_dir(data_dir, used_reservoir,
                                      used_band, time_steps)
    
    outputs = {}
    for data_type in data_types:
        outputs[data_type] = {}
        for file_type in file_types:
            outputs[data_type][file_type] = get_data_file_path(
                data_dir, used_reservoir, used_band,
                time_steps, data_type, file_type)
    
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
        _create_data_file(data_dir, used_reservoir, used_band, time_steps,
                          list_years[data_type], data_type, mask_data_dir)
    return outputs


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


def restore_data(path):
    """Restore cached data from disk to memory."""
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
    return data


def _get_data_path(data_file):
    with open(data_file, "r") as data_f:
        reader = csv.reader(data_f)
        data_paths = [row for row in reader]
    return data_paths


def _get_target_mask_path(target_file):
    with open(target_file, "r") as target_f:
        reader = csv.reader(target_f)
        target_paths = [row[0] for row in reader]
    return target_paths


def get_target_path(target_file):
    return _get_target_mask_path(target_file)[0]


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
            img = get_im(fl, reduce_size)
            currentX_.append(img)
        X_.append(currentX_)
    return X_


def _load_target_mask(target_file, reduce_size=None):
    target_mask_paths = _get_target_mask_path(target_file)
    y_ = []
    for path in target_mask_paths:
        img = get_im(path, reduce_size)
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


def get_data(data_dir='raw_data/MOD13Q1',
             used_reservoir=0,
             used_band='NDVI',
             time_steps=12,
             data_type='test',
             list_years=None,
             reduce_size=None,
             mask_data_dir='mask_data/MOD13Q1',
             force_recreated=False):
    """Get data, target and mask of a data type of a reservoir.
    
    Example:
        get_data(data_dir='raw_data/MOD13Q1',
                 used_reservoir=0,
                 used_band='NDVI',
                 time_steps=12,
                 data_type='test',
                 reduce_size=None,
                 mask_data_dir='mask_data/MOD13Q1',
                 force_recreated=False)
       
    Args:
        data_dir: String, directory where stores image data.
        used_reservoir: Integer, index of processed reservoir.
        used_band: String, a string represents name of used band.
        time_steps: Integer, time steps (length) of LSTM sequence.
        data_type: String, type of data (train/val/test).
        reduce_size: Tuple, desired size of loaded image (default is None).
        mask_data_dir: String, directory where stores masked image.
        force_recreated: Boolean, force recreated cache.

    Returns:
    Tuple of data, target and mask.
    """
    if list_years is None:
        list_years = get_list_years_default(data_type)

    dir_prefix = get_dir_prefix(used_reservoir, used_band, time_steps) 
    data_file_dir = get_data_file_dir(data_dir, used_reservoir, used_band,
                                      time_steps)
    cache_dir = get_cache_dir(data_dir, used_reservoir, used_band, time_steps)
    cache_path = os.path.join(cache_dir, '{}.dat'.format(data_type))

    target_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                     time_steps, data_type, 'target')
    if not force_recreated:
        if not os.path.isfile(target_file) \
                or not check_years_in_file(target_file, list_years):
            force_recreated = True

    if force_recreated or not os.path.isfile(cache_path):
        print('Read {} images.'.format(data_type))

        data_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                       time_steps, data_type, 'data')
        target_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                         time_steps, data_type, 'target')
        mask_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                       time_steps, data_type, 'mask')
        
        if force_recreated \
                or not os.path.isfile(data_file) \
                or not os.path.isfile(target_file) \
                or not os.path.isfile(mask_file):
            _create_data_file(data_dir, used_reservoir, used_band, time_steps,
                              list_years, data_type, mask_data_dir)

        data = _load_data(data_file, reduce_size=reduce_size)
        target = _load_target_mask(target_file, reduce_size=reduce_size)
        mask = _load_target_mask(mask_file, reduce_size=reduce_size)
        
        data = np.array(data, dtype=np.float32)
        data = np.expand_dims(data, axis=-1)
        target = np.array(target, dtype=np.float32)
        target = np.expand_dims(target, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        
        try:
            os.makedirs(cache_dir)
        except:
            pass

        cache = (data, target, mask)
        cache_data(cache, cache_path)
        return cache
    else:
        print('Restore {} from cache!'.format(data_type))
        cache = restore_data(cache_path)
        return cache


def normalize_data(data, mean=0.0, std=1.0):
    if std == 0:
        return data - mean
    return (data - mean) / std


def scale_data(data, min=-1000, max=10000, range_min=0, range_max=1):
    return np.interp(data, (min, max), (range_min, range_max))


def scale_data(data, range=(0,1)):
    return np.interp(data, (np.min(data), np.max(data)), range)


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
                  day=1, band_find='NIR'):
    dir = os.path.join(data_dir, str(reservoir_index), str(year),
                       str(year) + str(day).zfill(3))
    list_raster = os.listdir(dir)
    find = list(filter(lambda x: band_find in x, list_raster))
    return dir + '/' + find[0]


def find_img_name_1(data_dir='raw_data/MOD13Q1',
                    reservoir_index=0, year=2001,
                    day=1, band_find='NIR'):
    dir = os.path.join(data_dir, str(reservoir_index), str(year),
                       str(year) + str(day).zfill(3))
    list_raster = os.listdir(dir)
    find = list(filter(lambda x: band_find in x, list_raster))
    return find[0]
