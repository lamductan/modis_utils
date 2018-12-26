import numpy as np
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.models import Sequential, model_from_json, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D
from tensorflow.python.keras.layers import Conv1D, Conv2D
from tensorflow.python.keras.layers import GRU, LSTM
from tensorflow.python.keras.layers import ConvLSTM2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import losses

from modis_utils.misc import scale_data
from modis_utils.preprocessing.image_processing import mask_lake_img

def create_dir_prefix(time_steps, filters, kernel_size, 
                    n_hidden_layers, epochs=None):
    dir_prefix = os.path.join('time_steps_{}'.format(str(time_steps)), 
                              'filters_{}'.format(str(filters)), 
                              'kernel_size_{}'\
                                .format(str(kernel_size)),
                              'n_hidden_layers_{}'\
                                .format(str(n_hidden_layers)))
    if epochs is not None:
        dir_prefix = os.path.join(dir_prefix, 
                                  'epochs_{}'.format(str(epochs)))
    return dir_prefix


def save_model(model, time_steps, filters, kernel_size, 
               n_hidden_layers, epochs):
    
    dir_prefix = create_dir_prefix(time_steps, filters, kernel_size, 
                                   n_hidden_layers, epochs)
    try:
        json_string = model.to_json()
    except:
        json_string = None

    try:
        os.makedirs(os.path.join('cache', dir_prefix))
    except:
        pass

    if json_string:
        open(os.path.join(
                'cache', dir_prefix, 
                'architecture.json'), 
             'w'
            ).write(json_string)
   
    weight_path = os.path.join('cache', dir_prefix,
        'model_weights.h5')
    model.save(weight_path, overwrite=True)
    return weight_path


def load_model(time_steps, filters, kernel_size, 
               n_hidden_layers, epochs):
    dir_prefix = create_dir_prefix(time_steps, filters, kernel_size, 
                                 n_hidden_layers, epochs)
    architecture_path = os.path.join('cache', dir_prefix,
                                     'architecture.json')
    if os.path.isfile(architecture_path):
        model = model_from_json(open(architecture_path).read())
        model.load_weights(os.path.join('cache', dir_prefix,
                                        'model_weights.h5'))
        return model
    else:
        return load_model(os.path.join('cache', dir_prefix,
                                       'model_weights.h5'))


def load_latest_model(time_steps, filters, kernel_size, 
                      n_hidden_layers):
    dir_prefix = create_dir_prefix(time_steps, filters, kernel_size, 
                                 n_hidden_layers)
    list_dir = os.listdir(os.path.join('cache', dir_prefix))
    list_dir = filter(lambda x: '.h5' not in x, list_dir)
    epochs = [int(dir.split('_')[-1]) for dir in list_dir]
    max_epochs = max(epochs)
    return max_epochs


# Create model
def _create_model_with_tensorflow_1(model_params, compile_params):
    input_shape = model_params['input_shape']
    n_hidden_layers = 3

    filters = 16
    kernel_size = 5
    strides = (1, 1)
    padding = 'valid'
    data_format = None
    dilation_rate = (1, 1)
    activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'
    use_bias = True
    kernel_initializer = 'glorot_uniform'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer=None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    return_sequences = False
    go_backwards = False
    stateful = False
    dropout = 0.0
    recurrent_dropout = 0.0
    output_activation = 'sigmoid'

    if 'filters' in model_params.keys():
        filters = model_params['filters']
    if 'kernel_size' in model_params.keys():
        kernel_size = model_params['kernel_size']
    if 'strides' in model_params.keys():
        strides = model_params['strides']
    if 'padding' in model_params.keys():
        padding = model_params['padding']
    if 'data_format' in model_params.keys():
        data_format = model_params['data_format']
    if 'dilation_rate' in model_params.keys():
        dilation_rate = model_params['dilation_rate']

    if 'activation' in model_params.keys():
        activation = model_params['activation']
    if 'recurrent_activation' in model_params.keys():
        recurrent_activation = model_params['recurrent_activation']
    if 'use_bias' in model_params.keys():
        use_bias = model_params['use_bias']

    if 'kernel_initializer' in model_params.keys():
        kernel_initializer = model_params['kernel_initializer']
    if 'recurrent_initializer' in model_params.keys():
        recurrent_initializer = model_params['recurrent_initializer']
    if 'bias_initializer' in model_params.keys():
        bias_initializer = model_params['bias_initializer']
    if 'unit_forget_bias' in model_params.keys():
        unit_forget_bias = model_params['unit_forget_bias']
    if 'kernel_regularizer' in model_params.keys():
        kernel_regularizer = model_params['kernel_regularizer']
    if 'recurrent_regularizer' in model_params.keys():
        recurrent_regularizer = model_params['recurrent_regularizer']
    if 'bias_regularizer' in model_params.keys():
        bias_regularizer = model_params['bias_regularizer']
    if 'activity_regularizer' in model_params.keys():
        activity_regularizer = model_params['activity_regularizer']

    if 'kernel_constraint' in model_params.keys():
        kernel_constraint = model_params['kernel_constraint']
    if 'recurrent_constraint' in model_params.keys():
        recurrent_constraint = model_params['recurrent_constraint']
    if 'bias_constraint' in model_params.keys():
        bias_constraint = model_params['bias_constraint']
    if 'go_backwards' in model_params.keys():
        go_backwards = model_params['go_backwards']
    if 'stateful' in model_params.keys():
        stateful = model_params['stateful']
    if 'dropout' in model_params.keys():
        dropout = model_params['dropout']
    if 'recurrent_dropout' in model_params.keys():
        recurrent_dropout = model_params['recurrent_dropout']

    if 'output_activation' in model_params.keys():
        output_activation = model_params['output_activation']
    if 'n_hidden_layers' in model_params.keys():
        n_hidden_layers = model_params['n_hidden_layers']

    kernel_size_tuple = (kernel_size, kernel_size)

    source = keras.Input(
        name='seed', shape=input_shape, dtype=tf.float32)

    convLSTM_layers = [0]*(n_hidden_layers)
    batchNorm_layers = [0]*(n_hidden_layers)
 
    convLSTM_layers[0] = ConvLSTM2D(filters=filters, 
                                    kernel_size=kernel_size_tuple,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    dilation_rate=dilation_rate,
                                    activation=activation,
                                    recurrent_activation=recurrent_activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    unit_forget_bias=unit_forget_bias,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    return_sequences=True,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout)(source)
    batchNorm_layers[0] = BatchNormalization()(convLSTM_layers[0])

    for i in range(1, n_hidden_layers - 1):
        convLSTM_layers[i] = ConvLSTM2D(filters=filters, 
                                        kernel_size=kernel_size_tuple,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        dilation_rate=dilation_rate,
                                        activation=activation,
                                        recurrent_activation=recurrent_activation,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        recurrent_initializer=recurrent_initializer,
                                        bias_initializer=bias_initializer,
                                        unit_forget_bias=unit_forget_bias,
                                        kernel_regularizer=kernel_regularizer,
                                        recurrent_regularizer=recurrent_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        return_sequences=True,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout)(batchNorm_layers[i-1])

        batchNorm_layers[i] = BatchNormalization()(convLSTM_layers[i])
    
    convLSTM_layers[-1] = ConvLSTM2D(filters=filters, 
                                    kernel_size=kernel_size_tuple,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    dilation_rate=dilation_rate,
                                    activation=activation,
                                    recurrent_activation=recurrent_activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    unit_forget_bias=unit_forget_bias,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    return_sequences=False,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout)(batchNorm_layers[-2])

    batchNorm_layers[-1] = BatchNormalization()(convLSTM_layers[-1])

    predicted_img = Conv2D(filters=1, 
                           kernel_size=kernel_size_tuple,
                           strides=strides,
                           activation=output_activation,
                           padding=padding, 
                           data_format=data_format,
                           dilation_rate=dilation_rate,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint)(batchNorm_layers[-1])
    predicted_img = predicted_img.eval(tf.get_default_session())
    predicted_img = scale_data(predicted_img, (-1.0, 1.0), (-0.21, 1.0))
    mask_lake = mask_lake_img(predicted_img)
    mask_lake = tf.convert_to_tensor(mask_lake, np.int)
    model = keras.Model(inputs=[source], outputs=[mask_lake])

    # Compile parameters
    optimizer = keras.optimizers.SGD(lr=1e-4)
    loss = 'mse'
    metrics = ['mse']

    if 'optimizer' in compile_params.keys():
        optimizer = compile_params['optimizer']
    if 'loss' in compile_params.keys():
        loss = compile_params['loss']
    if 'metrics' in compile_params.keys():
        metrics= compile_params['metrics']

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    return model

def _create_model_with_tensorflow_2(model_params, compile_params):
    input_shape = model_params['input_shape']
    n_hidden_layers = 3

    filters = 16
    kernel_size = 5
    strides = (1, 1)
    padding = 'valid'
    data_format = None
    dilation_rate = (1, 1)
    activation = 'tanh'
    recurrent_activation = 'hard_sigmoid'
    use_bias = True
    kernel_initializer = 'glorot_uniform'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer=None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    return_sequences = False
    go_backwards = False
    stateful = False
    dropout = 0.0
    recurrent_dropout = 0.0
    output_activation = 'sigmoid'

    if 'filters' in model_params.keys():
        filters = model_params['filters']
    if 'kernel_size' in model_params.keys():
        kernel_size = model_params['kernel_size']
    if 'strides' in model_params.keys():
        strides = model_params['strides']
    if 'padding' in model_params.keys():
        padding = model_params['padding']
    if 'data_format' in model_params.keys():
        data_format = model_params['data_format']
    if 'dilation_rate' in model_params.keys():
        dilation_rate = model_params['dilation_rate']

    if 'activation' in model_params.keys():
        activation = model_params['activation']
    if 'recurrent_activation' in model_params.keys():
        recurrent_activation = model_params['recurrent_activation']
    if 'use_bias' in model_params.keys():
        use_bias = model_params['use_bias']

    if 'kernel_initializer' in model_params.keys():
        kernel_initializer = model_params['kernel_initializer']
    if 'recurrent_initializer' in model_params.keys():
        recurrent_initializer = model_params['recurrent_initializer']
    if 'bias_initializer' in model_params.keys():
        bias_initializer = model_params['bias_initializer']
    if 'unit_forget_bias' in model_params.keys():
        unit_forget_bias = model_params['unit_forget_bias']
    if 'kernel_regularizer' in model_params.keys():
        kernel_regularizer = model_params['kernel_regularizer']
    if 'recurrent_regularizer' in model_params.keys():
        recurrent_regularizer = model_params['recurrent_regularizer']
    if 'bias_regularizer' in model_params.keys():
        bias_regularizer = model_params['bias_regularizer']
    if 'activity_regularizer' in model_params.keys():
        activity_regularizer = model_params['activity_regularizer']

    if 'kernel_constraint' in model_params.keys():
        kernel_constraint = model_params['kernel_constraint']
    if 'recurrent_constraint' in model_params.keys():
        recurrent_constraint = model_params['recurrent_constraint']
    if 'bias_constraint' in model_params.keys():
        bias_constraint = model_params['bias_constraint']
    if 'go_backwards' in model_params.keys():
        go_backwards = model_params['go_backwards']
    if 'stateful' in model_params.keys():
        stateful = model_params['stateful']
    if 'dropout' in model_params.keys():
        dropout = model_params['dropout']
    if 'recurrent_dropout' in model_params.keys():
        recurrent_dropout = model_params['recurrent_dropout']

    if 'output_activation' in model_params.keys():
        output_activation = model_params['output_activation']
    if 'n_hidden_layers' in model_params.keys():
        n_hidden_layers = model_params['n_hidden_layers']

    kernel_size_tuple = (kernel_size, kernel_size)

    source = keras.Input(
        name='seed', shape=input_shape, dtype=tf.float32)

    convLSTM_layers = [0]*(n_hidden_layers)
    batchNorm_layers = [0]*(n_hidden_layers)
 
    convLSTM_layers[0] = ConvLSTM2D(filters=filters[0], 
                                    kernel_size=kernel_size_tuple,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    dilation_rate=dilation_rate,
                                    activation=activation,
                                    recurrent_activation=recurrent_activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    unit_forget_bias=unit_forget_bias,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    return_sequences=True,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout)(source)
    batchNorm_layers[0] = BatchNormalization()(convLSTM_layers[0])

    for i in range(1, n_hidden_layers - 1):
        convLSTM_layers[i] = ConvLSTM2D(filters=filters[i], 
                                        kernel_size=kernel_size_tuple,
                                        strides=strides,
                                        padding=padding,
                                        data_format=data_format,
                                        dilation_rate=dilation_rate,
                                        activation=activation,
                                        recurrent_activation=recurrent_activation,
                                        use_bias=use_bias,
                                        kernel_initializer=kernel_initializer,
                                        recurrent_initializer=recurrent_initializer,
                                        bias_initializer=bias_initializer,
                                        unit_forget_bias=unit_forget_bias,
                                        kernel_regularizer=kernel_regularizer,
                                        recurrent_regularizer=recurrent_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        return_sequences=True,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout)(batchNorm_layers[i-1])

        batchNorm_layers[i] = BatchNormalization()(convLSTM_layers[i])
    
    convLSTM_layers[-1] = ConvLSTM2D(filters=filters[-1], 
                                    kernel_size=kernel_size_tuple,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    dilation_rate=dilation_rate,
                                    activation=activation,
                                    recurrent_activation=recurrent_activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    unit_forget_bias=unit_forget_bias,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    return_sequences=False,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout)(batchNorm_layers[-2])

    batchNorm_layers[-1] = BatchNormalization()(convLSTM_layers[-1])

    predicted_img = Conv2D(filters=1,
                           kernel_size=kernel_size_tuple,
                           strides=strides,
                           activation=output_activation,
                           padding=padding, 
                           data_format=data_format,
                           dilation_rate=dilation_rate,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint)(batchNorm_layers[-1])
    predicted_img = predicted_img.eval(tf.get_default_session())
    predicted_img = scale_data(predicted_img, (-1.0, 1.0), (-0.21, 1.0))
    mask_lake = mask_lake_img(predicted_img)
    mask_lake = tf.convert_to_tensor(mask_lake, np.int)
    model = keras.Model(inputs=[source], outputs=[mask_lake])

    # Compile parameters
    optimizer = keras.optimizers.SGD(lr=1e-4)
    loss = 'mse'
    metrics = ['mse']

    if 'optimizer' in compile_params.keys():
        optimizer = compile_params['optimizer']
    if 'loss' in compile_params.keys():
        loss = compile_params['loss']
    if 'metrics' in compile_params.keys():
        metrics= compile_params['metrics']

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    return model


def create_model_with_tensorflow(model_params, compile_params):
    if not isinstance(model_params['filters'], list):
        return _create_model_with_tensorflow_1(model_params, compile_params)
    else:
        return _create_model_with_tensorflow_2(model_params, compile_params)
