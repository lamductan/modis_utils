import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from modis_utils.model.core import create_model_with_tensorflow
from modis_utils.misc import get_data, scale_data, restore_data
from modis_utils.misc import get_cache_file_path
from modis_utils.misc import scale_normalized_data
from modis_utils.preprocessing.image_processing import mask_lake_img 

ORIGINAL_RANGE = {'NDVI': (-2000, 10000)}

def create_dir_prefix(time_steps, filters, kernel_size, 
                      n_hidden_layers, epochs=None):
    dir_prefix = os.path.join(
        'time_steps_{}'.format(str(time_steps)),
        'filters_{}'.format(str(filters)),
        'kernel_size_{}'.format(str(kernel_size)),
        'n_hidden_layers_{}'.format(str(n_hidden_layers))
    )
    if epochs is not None:
        dir_prefix = os.path.join(dir_prefix, 
                                  'epochs_{}'.format(str(epochs)))
    return dir_prefix


def predict_and_visualize(data, target, model,
                          which=0, result_dir=None):
    time_steps = data.shape[1]
    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, time_steps)
    
    example = data[which, :, :, :,0]
    for i, img in enumerate(example):
        axe = plt.subplot(G[0, i])
        axe.imshow(img)

    target_example = target[which, :, :, 0]
    pred = model.predict(data[which][np.newaxis, :, :, :, :])
    del model

    ax_groundtruth = plt.subplot(G[1, :time_steps//2])
    ax_groundtruth.imshow(target_example)
    ax_groundtruth.set_title('groundtruth')
    
    ax_pred = plt.subplot(G[1, time_steps//2:2*(time_steps//2)])
    ax_pred.imshow(pred[0, :, :, 0])
    ax_pred.set_title('predict')

    if result_dir is not None:
        #eval = model.evaluate(np.expand_dims(data[which], axis=0), 
        #                      np.expand_dims(target[which], axis=0))
        try:
            os.makedirs(result_dir)
        except:
            pass

        #with open(os.path.join(result_dir, 'log.txt'), 'a') as w:
        #    w.write('{},{}'.format(eval[0], eval[1]))
        #    w.write('\n')

        plt.savefig(os.path.join(result_dir, '{}.png'.format(which))) 

    return target_example, pred[0, :, :, 0]


def predict_and_visualize_RandomCrop(data, target, model, crop_size,
                                     which=0, result_dir=None):
    time_steps = data.shape[1]
    input_seq = data[which]
    ground_truth = target[which]
    
    offset_x = data.shape[2] % crop_size
    offset_y = data.shape[3] % crop_size
    input_seq = input_seq[:, offset_x//2:-(offset_x - offset_x//2), \
                          offset_y//2:-(offset_y - offset_y//2), :]
    ground_truth = ground_truth[offset_x//2:-(offset_x - offset_x//2), \
                                offset_y//2:-(offset_y - offset_y//2), :]

    predict = np.zeros_like(ground_truth)

    for i in range(input_seq.shape[1] // crop_size):
        for j in range(input_seq.shape[2] // crop_size):
            pred = model.predict(input_seq[np.newaxis, :, \
                                 i*crop_size:(i+1)*crop_size, \
                                 j*crop_size:(j+1)*crop_size, :])
            predict[i*crop_size:(i+1)*crop_size, \
                    j*crop_size:(j+1)*crop_size, :] = pred[0]

    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, time_steps)

    for i, img in enumerate(input_seq[:, :, :, 0]):
        axe = plt.subplot(G[0, i])
        axe.imshow(img)

    ax_groundtruth = plt.subplot(G[1, :time_steps//2])
    ax_groundtruth.imshow(ground_truth[:, :, 0])
    ax_groundtruth.set_title('groundtruth')

    ax_pred = plt.subplot(G[1, time_steps//2:2*(time_steps//2)])
    ax_pred.imshow(predict[:, :, 0])
    ax_pred.set_title('predict') 

    if result_dir is not None:
        try:
            os.makedirs(result_dir)
        except:
            pass
        plt.savefig(os.path.join(result_dir, '{}.png'.format(which))) 

    return ground_truth[:,:,0], predict[:,:,0]
    

def test_by_gridding(reservoir_index,
      	             weight_path='weights.h5',
             	     crop_size=16,
                     time_steps=8,
                     filters=16,
                     kernel_size=1,
                     n_hidden_layers=3,
                     model=None,
                     which=None,
                     result_dir=None):
    test_data, test_target = get_data(data_type='test', 
                                      reservoir_index=reservoir_index, 
                                      time_steps=time_steps)
    test_data = scale_data(test_data)
    test_target = scale_data(test_target)
    
    input_shape = (time_steps, crop_size, crop_size, 1) 
    
    if model is None:
        model = create_model_with_tensorflow(filters, kernel_size, 
                                             input_shape, n_hidden_layers)
        model.load_weights(weight_path)
    
    n = test_data.shape[0]
    if which is None:
        which=np.random.randint(n)
    return predict_and_visualize_RandomCrop(which=which,
                                            reservoir_index=reservoir_index,
                                            test_data=test_data,
                                            test_target=test_target,
                                            model=model,
                                            crop_size=crop_size,
                                            result_dir=result_dir)


def test_not_gridding(reservoir_index,
             	      weight_path='weights.h5',
            	      crop_size=16,
                      time_steps=8,
                      filters=16,
                      kernel_size=1,
                      n_hidden_layers=3,
                      model=None,
                      which=None,
                      result_dir=None):
    test_data, test_target = get_data(data_type='test', 
                                      reservoir_index=reservoir_index, 
                                      time_steps=time_steps)
    test_data = scale_data(test_data)
    test_target = scale_data(test_target)

    n = test_data.shape[0]
    img_col = test_target.shape[1]
    img_row = test_target.shape[2]

    input_shape = (time_steps, img_col, img_row, 1) 
    if model is None:
        model = create_model_with_tensorflow(filters, kernel_size,
                                             input_shape, n_hidden_layers)
        model.load_weights(weight_path)

    n = test_data.shape[0]
    if which is None:
        which=np.random.randint(n)
        
    input_seq = test_data[which]
    ground_truth = test_target[which]
    pred = model.predict(input_seq[np.newaxis, :, :, :, :])
    
    plt.figure(figsize=(10, 10))
    G = gridspec.GridSpec(2, time_steps)

    for i, img in enumerate(input_seq[:, :, :, 0]):
        axe = plt.subplot(G[0, i])
        axe.imshow(img)

    ax_groundtruth = plt.subplot(G[1, :time_steps//2])
    ax_groundtruth.imshow(ground_truth[:, :, 0])
    ax_groundtruth.set_title('groundtruth')

    ax_pred = plt.subplot(G[1, time_steps//2:2*(time_steps//2)])
    ax_pred.imshow(pred[0, :, :, 0])
    ax_pred.set_title('predict') 

    if result_dir is None:
        dir_prefix = os.path.join('results_0', 'random_crop',
                 'crop_size_{}'.format(crop_size), str(reservoir_index))
        try:
            os.makedirs(dir_prefix)
        except:
            pass
        result_dir = dir_prefix
    plt.savefig(os.path.join(result_dir, '{}.png'.format(which)))
    
    return ground_truth[:,:,0], pred[0,:,:,0] 


def visualize(model, input, groundtruth):
    fig, axes = plt.subplots(1, 2, figsize=(10,10))
    axes[0].imshow(groundtruth[:,:,0])
    axes[0].set_title('groundtruth')
  
    predict = model.predict(input[np.newaxis,:,:,:,:])
    axes[1].imshow(predict[0,:,:,0])
    axes[1].set_title('predict')
    return groundtruth[:,:,0], predict[0,:,:,0]


def mse_with_mask_tf(y_true_and_mask, y_pred):
    y_true, y_mask = tf.split(y_true_and_mask, 2, axis=-1)
    square_error = (y_true - y_pred)**2
    tf_mask = tf.where(tf.equal(y_mask, -1),
                       tf.fill(tf.shape(y_mask), 0),
                       tf.fill(tf.shape(y_mask), 1))
    tf_mask = tf.to_float(tf_mask)
    #return tf.reduce_mean(tf.multiply(tf_mask, square_error))
    return tf.divide(tf.reduce_sum(tf.multiply(tf_mask, square_error)),
                     tf.maximum(tf.reduce_sum(tf_mask), 1.0))


def mse_with_mask(groundtruth, mask, predict):
    square_error = ((groundtruth - predict)**2)
    cloud_mask = np.where(mask == -1, 0.0, 1.0)
    return np.sum(np.multiply(cloud_mask, square_error))/np.maximum(
        np.sum(cloud_mask), 1.0)

  
def histogram(groundtruth, predict, bins=10, range=(0.0, 1.0)):
    groundtruth_hist = np.histogram(groundtruth, bins=bins, 
                                    range=range)
    predict_hist = np.histogram(predict, bins=bins, range=range)

    x = np.arange(bins)
    #ticks = np.linspace(range[0], range[1], bins + 1)[:-1]
    ticks = np.arange(-bins, bins, 2)/bins
    #print(ticks)

    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].bar(x, groundtruth_hist[0])
    ax[0].set_title('groundtruth hist')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(ticks)

    ax[1].bar(x, predict_hist[0])
    ax[1].set_title('prediction_hist')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(ticks)

    plt.show()


def test(reservoir_index, test_index, data_dir, used_band, time_steps,
         model_params_path, weight_path, result_dir_prefix=None):
    test_cache_path = get_cache_file_path(data_dir, reservoir_index, used_band,
                                          time_steps, 'test')
    if result_dir_prefix is None:
        result_dir = None
    else:
        result_dir = os.path.join(result_dir_prefix, str(reservoir_index))
    
    test_data, test_target, mask = restore_data(test_cache_path)
    mask = mask.squeeze(axis=-1)

    img_col = test_target.shape[1]
    img_row = test_target.shape[2] 
    input_shape = (time_steps, img_col, img_row, 1)
    model_params_non_gridding, compile_params = restore_data(model_params_path)
    model_params_non_gridding['input_shape'] = input_shape

    if compile_params['loss'] == 'mse_with_mask_tf':
        compile_params['loss'] = mse_with_mask_tf
    else:
        compile_params['loss'] = 'mse'
    if compile_params['metrics'] == ['mse_with_mast_tf']:
        compile_params['metrics'] = [mse_with_mast_tf]
    else:
        compile_params['metrics'] = ['mse']

    model_non_gridding = create_model_with_tensorflow(model_params_non_gridding,
                                                      compile_params)
    if model_params_non_gridding['output_activation'] == 'tanh':
        groundtruth_range = (-1.0, 1.0)
    else:
        groundtruth_range = (0.0, 1.0)
    test_target = scale_normalized_data(test_target, range=groundtruth_range)

    model_non_gridding.load_weights(weight_path)

    try:
        os.makedirs(result_dir)
    except:
        pass
    groundtruth, predict = predict_and_visualize(data=test_data,
                                                 target=test_target,
                                                 which=test_index,
                                                 model=model_non_gridding,
                                                 result_dir=result_dir)
    if result_dir is not None:
        log_path = os.path.join(result_dir, 'log.txt')
        metric = mse_with_mask(groundtruth, mask[test_index], predict)
        with open(log_path, 'a') as f:
            f.write('{:02} - {:04f}'.format(test_index, metric))
            f.write('\n')
        f.close()
    return groundtruth, predict, mask[test_index]


def to_binary_scale_img(scale_img,
                        used_band='NDVI',
                        range=(-1.0, 1.0)):
    original_range = ORIGINAL_RANGE[used_band]
    a = (scale_img - range[0])/(range[1] - range[0])
    a = a*(original_range[1] - original_range[0]) + original_range[0]
    return mask_lake_img(a, used_band)
