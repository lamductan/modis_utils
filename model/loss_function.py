import numpy as np
import tensorflow as tf

def mse_with_mask_tf(y_true_and_mask, y_pred):
    y_true, y_mask = tf.split(y_true_and_mask, 2, axis=-1)
    square_error = (y_true - y_pred)**2
    tf_mask = tf.where(tf.equal(y_mask, 0), #0
                       tf.fill(tf.shape(y_mask), 0),
                       tf.fill(tf.shape(y_mask), 1))
    tf_mask = tf.to_float(tf_mask)
    #return tf.reduce_mean(tf.multiply(tf_mask, square_error))
    return tf.divide(tf.reduce_sum(tf.multiply(tf_mask, square_error)),
                     tf.maximum(tf.reduce_sum(tf_mask), 1.0))


def mse_with_mask(groundtruth, mask, predict, mask_cloud=0):
    square_error = ((groundtruth - predict)**2)
    cloud_mask = np.where(mask == mask_cloud, 0.0, 1.0) #0
    return np.sum(np.multiply(cloud_mask, square_error))/np.maximum(
        np.sum(cloud_mask), 1.0)

