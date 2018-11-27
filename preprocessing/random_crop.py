import os
import numpy as np
import rasterio as rio
from scipy import misc
import multiprocessing as mp

from modis_utils.preprocessing.generators.generators import SimpleImageGenerator
from modis_utils.misc import create_data_file, get_data, scale_data
from modis_utils.misc import create_data_file_continuous_years
from modis_utils.misc import cache_data, restore_data
from modis_utils.misc import get_dir_prefix, get_data_augment_dir
from modis_utils.misc import get_data_file_path
from modis_utils.misc import get_list_years_default
from modis_utils.misc import get_data_augment_merged_dir
from modis_utils.misc import get_data_paths, get_target_paths, get_mask_paths
from modis_utils.misc import get_data_merged_from_paths

def _merge_data_target_mask(data, target, mask):
    target_1 = target.reshape(target.shape[0], 1, target.shape[1], 
                              target.shape[2], target.shape[3])
    mask_1 = mask.reshape(mask.shape[0], 1, mask.shape[1], 
                          mask.shape[2], mask.shape[3])
    data_1 = np.concatenate([data, target_1, mask_1], axis=1)
    data_1 = data_1.squeeze(axis=-1)
    return data_1


def _generate(data_merged, datagen, data_dir, used_reservoir, used_band,
              time_steps, list_years, data_type, mask_data_dir, n_samples,
              crop_size):
    data_iterator = datagen.flow_from_list(x=data_merged,
                                           nframes=data_merged.shape[1])
    cnt = 0
    for i in range(n_samples):
        batch = data_iterator._get_batches_of_transformed_samples(
                np.arange(data_merged.shape[0]))
        for j in range(batch.shape[0]):
            cur = batch[j]
            data = cur[:-2]
            data = np.expand_dims(data, axis=-1)
            target = np.expand_dims(cur[-2:-1], axis=-1)
            mask = np.expand_dims(cur[-1:], axis=-1)
            target_mask = np.concatenate((target, mask), axis=-1)
            data_augment_dir = get_data_augment_dir(data_dir,
                                                    used_reservoir,
                                                    used_band,
                                                    time_steps,
                                                    data_type,
                                                    crop_size)
            if not os.path.isdir(data_augment_dir):
                os.makedirs(data_augment_dir)
            file_path = os.path.join(data_augment_dir,
                                     '{}.dat'.format(cnt))
            cnt += 1
            cache_data((data, target_mask), file_path)
        del batch, cur, data, target, mask, target_mask


def _check_image_size(img_path, crop_size):
    img = restore_data(img_path)
    return (img.shape[0] > 3*crop_size and img.shape[1] > 3*crop_size)


def augment_one_reservoir(data_dir='raw/MOD13Q1',
                          used_reservoir=0,
                          used_band='NDVI',
                          time_steps=12,
                          crop_size=32,
                          random_crop=True,
                          train_list_years=None,
                          val_list_years=None,
                          test_list_years=None,
                          mask_data_dir='mask_data/MOD13Q1',
                          n_samples=100,
                          recreated_data_file=False):
    """Generate random crop augmentation on reservoir.

    Example:
        augment_one_reservoir(data_dir='raw_data/MOD13Q1',
                              used_reservoir=5,
                              used_band='NDVI',
                              time_steps=12,
                              crop_size=32,
                              random_crop=True,
                              train_list_years=None,
                              val_list_years=None,
                              test_list_years=None,
                              mask_data_dir='mask_data/MOD13Q1',
                              n_samples=100)

    Args:
        data_dir: String, directory where stores image data.
        used_reservoir: Integer, index of processed reservoir.
        used_band: String, a string represents name of used band.
        time_steps: Integer, time steps (length) of LSTM sequence.
        crop_size: Integer, size of crop.
        random_crop: Boolean
        train_list_years: List of integers, list years of train set.
        val_list_years: List of integers, list years of val set.
        test_list_years: List of integers, list years of test set.
        mask_data_dir: String, directory where stores masked image.
        n_samples: Integer, number of crop sample on each image.

    Returns:
        Boolean, True if augment successfully and False vice versa.
    """
    data_types = ['train', 'val', 'test']
    list_years = {}
    if train_list_years is None \
            or val_list_years is None \
            or test_list_years is None:
        for data_type in data_types:
            list_years[data_type] = get_list_years_default(data_type)
    else:
        list_years['train'] = train_list_years
        list_years['val'] = val_list_years
        list_years['test'] = test_list_years

    if recreated_data_file:
        create_data_file(data_dir=data_dir,
                         used_reservoir=used_reservoir,
                         used_band=used_band,
                         time_steps=time_steps,
                         train_list_years=list_years['train'],
                         val_list_years=list_years['val'],
                         test_list_years=list_years['test'],
                         mask_data_dir=mask_data_dir)

    sample_target_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                     time_steps, 'train', 'target')
    sample_img_path = get_target_path(sample_target_file)[0]
    if not _check_image_size(sample_img_path, crop_size):
        return False

    datagen = SimpleImageGenerator(crop_size=(crop_size, crop_size),
                                   random_crop=random_crop)
    for data_type in data_types:
        (data, target, mask) = get_data(data_dir=data_dir,
                                        used_reservoir=used_reservoir,
                                        used_band=used_band,
                                        time_steps=time_steps,
                                        list_years=list_years[data_type],
                                        data_type=data_type,
                                        mask_data_dir=mask_data_dir)
        data_merged = _merge_data_target_mask(data, target, mask)
        del data, target, mask
        _generate(data_merged, datagen, data_dir, used_reservoir, used_band,
                  time_steps, list_years[data_type], data_type, mask_data_dir,
                  n_samples, crop_size)
        del data_merged
    return True


def _merge_data_augment(n_data, data_index_shuffle, list_data, merge_data_dir,
                        batch_size, thread_id, n_threads):
    m = n_data//batch_size
    k = m//n_threads
    i_range_of_cur_thread = range(thread_id*k, (thread_id+1)*k)
    for i in i_range_of_cur_thread:
        merge_data = []
        merge_target_mask = []
        for j in data_index_shuffle[i*batch_size : (i+1)*batch_size]:
            data = restore_data(list_data[j])
            merge_data.append(np.expand_dims(data[0], axis=0))
            merge_target_mask.append(data[1])
        if len(merge_data) == batch_size:
            merge_data = np.vstack(merge_data)
            merge_target_mask = np.vstack(merge_target_mask)
            merge_data_path = os.path.join(merge_data_dir, '{}.dat'.format(i))
            cache_data((merge_data, merge_target_mask), merge_data_path)
        

def _merge_last_data_augment(n_data, data_index_shuffle, list_data, merge_data_dir, 
                             batch_size, thread_id, n_threads):
    m = n_data//batch_size
    k = m - m % n_threads
    i = k + thread_id
    merge_data = []
    merge_target_mask = []
    for j in data_index_shuffle[i*batch_size : (i+1)*batch_size]:
        data = restore_data(list_data[j])
        merge_data.append(np.expand_dims(data[0], axis=0))
        merge_target_mask.append(data[1])
    if len(merge_data) == batch_size:
        merge_data = np.vstack(merge_data)
        merge_target_mask = np.vstack(merge_target_mask)
        merge_data_path = os.path.join(merge_data_dir, '{}.dat'.format(i))
        cache_data((merge_data, merge_target_mask), merge_data_path)
    

def _merge_data_with_last(merge_data_dir, n_data, data_index_shuffle, list_data, 
                          batch_size, n_threads):
    try:
        os.makedirs(merge_data_dir)
    except:
        pass
    
    processes = [mp.Process(target=_merge_data_augment, 
                            args=(n_data, data_index_shuffle, list_data,
                                  merge_data_dir, batch_size, thread_id, 
                                  n_threads))
                 for thread_id in range(n_threads)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    processes = [mp.Process(target=_merge_last_data_augment,
                            args=(n_data, data_index_shuffle, list_data, 
                                  merge_data_dir, batch_size, thread_id,
                                  n_threads))
                 for thread_id in range(n_threads)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def merge_data_augment_train_val(data_dir,
                                 used_band,
                                 time_steps,
                                 crop_size,
                                 batch_size=1024,
                                 n_threads=mp.cpu_count() - 2):
    """Merge multiple crop files for speed up uploading and training
    on Google Colab."

    Example:
        merge_data_augment_train_val(data_dir='raw_data/MOD13Q1',
                                     used_band='NDVI',
                                     time_steps=12,
                                     crop_size=32,
                                     batch_size=1024,
                                     n_threads=6)

    Args:
        data_dir: String, directory where stores image data.
        used_band: String, a string represents name of used band.
        time_steps: Integer, time steps (length) of LSTM sequence.
        crop_size: Integer, size of crop.
        batch_size: Integer, number of crop files merged to one unique file.
        n_threads: Integer, number of parallel threads. It is recommended to
            set this parameter equal to your core number minus 2.
    """
    data_augment_dir = os.path.join('data_augment', data_dir,
                                    str(crop_size), str(time_steps))
    list_used_reservoirs = list(map(int, os.listdir(data_augment_dir)))
    list_train_data = []
    list_val_data = []
    for used_reservoir in list_used_reservoirs:
        train_path = get_data_augment_dir(data_dir, used_reservoir, used_band,
                                          time_steps, 'train', crop_size)
        val_path = get_data_augment_dir(data_dir, used_reservoir, used_band,
                                        time_steps, 'val', crop_size)
        list_train_data += [os.path.join(train_path, data_index) for data_index 
                            in os.listdir(train_path)]
        list_val_data += [os.path.join(val_path, data_index) for data_index 
                          in os.listdir(val_path)]
    n_train = len(list_train_data)
    n_val = len(list_val_data)
    train_index_shuffle = np.random.permutation(n_train)
    val_index_shuffle = np.random.permutation(n_val)
    
    merge_data_train_dir = get_data_augment_merged_dir(data_dir, used_band,
                                            time_steps, 'train', crop_size)
    merge_data_val_dir = get_data_augment_merged_dir(data_dir, used_band,
                                            time_steps, 'val', crop_size)
    _merge_data_with_last(merge_data_train_dir, n_train, train_index_shuffle,
                          list_train_data, batch_size, n_threads)
    _merge_data_with_last(merge_data_val_dir, n_val, val_index_shuffle,
                          list_val_data, batch_size, n_threads)


def augment_one_reservoir_without_cache(data_dir='raw/MOD13Q1',
                                        used_reservoir=0,
                                        used_band='NDVI',
                                        time_steps=12,
                                        crop_size=32,
                                        random_crop=True,
                                        train_list_years=None,
                                        val_list_years=None,
                                        test_list_years=None,
                                        mask_data_dir='mask_data/MOD13Q1',
                                        n_samples=100,
                                        recreated_data_file=False):
    """Generate random crop augmentation on reservoir.

    Example:
        augment_one_reservoir_without_cache(data_dir='raw_data/MOD13Q1',
                                            used_reservoir=5,
                                            used_band='NDVI',
                                            time_steps=12,
                                            crop_size=32,
                                            random_crop=True,
                                            train_list_years=None,
                                            val_list_years=None,
                                            test_list_years=None,
                                            mask_data_dir='mask_data/MOD13Q1',
                                            n_samples=100,
                                            recreated_data_file=False)

    Args:
        data_dir: String, directory where stores image data.
        used_reservoir: Integer, index of processed reservoir.
        used_band: String, a string represents name of used band.
        time_steps: Integer, time steps (length) of LSTM sequence.
        crop_size: Integer, size of crop.
        random_crop: Boolean
        train_list_years: List of integers, list years of train set.
        val_list_years: List of integers, list years of val set.
        test_list_years: List of integers, list years of test set.
        mask_data_dir: String, directory where stores masked image.
        n_samples: Integer, number of crop sample on each image.

    Returns:
        Boolean, True if augment successfully and False vice versa.
    """
    data_types = ['train', 'val', 'test']
    list_years = {}
    if train_list_years is None \
            or val_list_years is None \
            or test_list_years is None:
        for data_type in data_types:
            list_years[data_type] = get_list_years_default(data_type)
    else:
        list_years['train'] = train_list_years
        list_years['val'] = val_list_years
        list_years['test'] = test_list_years

    if recreated_data_file:
        create_data_file_continuous_years(data_dir=data_dir,
                                          used_reservoir=used_reservoir,
                                          used_band=used_band,
                                          time_steps=time_steps,
                                          train_list_years=list_years['train'],
                                          val_list_years=list_years['val'],
                                          test_list_years=list_years['test'],
                                          mask_data_dir=mask_data_dir)

    sample_target_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                     time_steps, 'train', 'target')
    sample_img_path = get_target_paths(sample_target_file)[0]
    if not _check_image_size(sample_img_path, crop_size):
        return False

    datagen = SimpleImageGenerator(crop_size=(crop_size, crop_size),
                                   random_crop=random_crop)
    for data_type in data_types:
        data_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                       time_steps, data_type, 'data')
        data_paths = get_data_paths(data_file)

        target_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                         time_steps, data_type, 'target')
        target_paths = get_target_paths(target_file)

        mask_file = get_data_file_path(data_dir, used_reservoir, used_band,
                                       time_steps, data_type, 'mask')
        mask_paths = get_mask_paths(mask_file)

        _generate_without_cache(data_paths, target_paths, mask_paths,
                                datagen, data_dir, used_reservoir, used_band, time_steps,
                                list_years[data_type], data_type, mask_data_dir,
                                n_samples, crop_size)
    return True


def _generate_without_cache(data_paths, target_paths, mask_paths,
                            datagen, data_dir, used_reservoir, used_band, time_steps,
                            list_years, data_type, mask_data_dir,
                            n_samples, crop_size):
    n_data = len(data_paths)
    cnt = 0
    for k in range(n_data):
        data_merged = get_data_merged_from_paths(data_paths[k], target_paths[k],
                                                 mask_paths[k])
        data_iterator = datagen.flow_from_list(x=data_merged,
                                               nframes=data_merged.shape[1])
        for i in range(n_samples):
            batch = data_iterator._get_batches_of_transformed_samples(
                    np.arange(data_merged.shape[0]))
            for j in range(batch.shape[0]):
                cur = batch[j]
                data = cur[:-2]
                data = np.expand_dims(data, axis=-1)
                target = np.expand_dims(cur[-2:-1], axis=-1)
                mask = np.expand_dims(cur[-1:], axis=-1)
                target_mask = np.concatenate((target, mask), axis=-1)
                data_augment_dir = get_data_augment_dir(data_dir,
                                                        used_reservoir,
                                                        used_band,
                                                        time_steps,
                                                        data_type,
                                                        crop_size)
                if not os.path.isdir(data_augment_dir):
                    os.makedirs(data_augment_dir)
                file_path = os.path.join(data_augment_dir,
                                         '{}.dat'.format(cnt))
                cnt += 1
                cache_data((data, target_mask), file_path)
            del batch, cur, data, target, mask, target_mask

