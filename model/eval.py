import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from modis_utils.model.core import create_model_with_tensorflow
from modis_utils.misc import get_data, scale_data

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
    
    ax_groundtruth = plt.subplot(G[1, :time_steps//2])
    ax_groundtruth.imshow(target_example)
    ax_groundtruth.set_title('groundtruth')
    
    ax_pred = plt.subplot(G[1, time_steps//2:2*(time_steps//2)])
    ax_pred.imshow(pred[0, :, :, 0])
    ax_pred.set_title('predict')

    if result_dir is not None:
        eval = model.evaluate(np.expand_dims(data[which], axis=0), 
                              np.expand_dims(target[which], axis=0))
        try:
            os.makedirs(result_dir)
        except:
            pass

        with open(os.path.join(result_dir, 'log.txt'), 'a') as w:
            w.write('{},{}'.format(eval[0], eval[1]))
            w.write('\n')

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
