import os
from scipy import misc
from shutil import make_archive
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler, CSVLogger

from modis_utils.preprocessing.preprocess_strategy_context import PreprocessStrategyContext
from modis_utils.image_processing import create_water_cloud_mask, change_fill_value
from modis_utils.misc import create_data_file_continuous_years, get_data_file_path
from modis_utils.preprocessing.random_crop import augment_one_reservoir_without_cache
from modis_utils.preprocessing.random_crop import merge_data_augment
from modis_utils.misc import get_data_test, get_target_test, cache_data, restore_data, get_data_paths
from modis_utils.generators import get_generator
from modis_utils.model.core import create_model_by_name, compile_model
from modis_utils.model.loss_function import PSNRLoss, lossSSIM, SSIM, step_decay
from modis_utils.model.loss_function import mse_with_mask_tf, mse_with_mask_tf_1, mse_with_mask


class ModisUtils:

    def __init__(self,
                 raw_data_dir='../raw_data',
                 modis_product='MOD13Q1',
                 reservoir_index=0,
                 preprocessed_type='normalized_div',
                 used_band='NDVI',
                 crop_size=32,
                 input_timesteps=12,
                 output_timesteps=1,
                 year_range=(2000, 2018),
                 model_name='convlstm_simple',
                 batch_size=32,
                 model_keras=True,
                 compile_params=None,
                 original_batch_size=1024,
                 TPU_FLAG=False,
                 training=True,
                 monitor=None,
                 monitor_mode='min'):
        # Define parameters
        self._modis_product = modis_product
        self._reservoir_index = reservoir_index
        self._raw_data_dir = os.path.join(
            raw_data_dir, self._modis_product,
            str(self._reservoir_index))
        self._preprocessed_type = preprocessed_type
        self._used_band = used_band
        self._crop_size = crop_size
        self._input_timesteps = input_timesteps
        self._output_timesteps = output_timesteps
        self._year_range = year_range
        self._original_batch_size = original_batch_size
        self._TPU_FLAG = TPU_FLAG
        
        # Dir and Dir prefix
        self._dir_prefix = os.path.join(
            self._modis_product, str(self._reservoir_index), 
            self._used_band, self._preprocessed_type)
        
        self._preprocessed_data_dir_prefix = os.path.join(
            'preprocessed_data', self._modis_product, 
            str(self._reservoir_index), self._used_band)
        
        self._preprocessed_data_dir = os.path.join(
            self._preprocessed_data_dir_prefix, self._preprocessed_type)
        
        self._mask_data_dir = os.path.join(
            'masked_data', self._modis_product, 
            str(self._reservoir_index), self._used_band)
        
        self._data_augment_dir_prefix = os.path.join(
            self._dir_prefix, str(self._input_timesteps), 
            str(self._output_timesteps), str(self._crop_size))
        self._data_augment_dir = os.path.join(
            'data_augment', self._data_augment_dir_prefix)
        self._data_augment_merged_dir = os.path.join(
            'data_augment_merged', self._data_augment_dir_prefix)
        
        # Other parameters
        self._day_period = 8 if self._modis_product == 'ALL' else 16
        self._n_data_per_year = 365//self._day_period + 1
        self._list_years = list(range(year_range[0], year_range[1] + 1))
        self._list_years_train = self._list_years[:-7]
        self._list_years_val = self._list_years[-7:-4]
        self._list_years_test = self._list_years[-4:]
    
        self._data_files = self._get_data_files()
        
        # Model parameters
        self._batch_size = batch_size
        self._model_name = model_name
        self._train_filenames = None
        self._val_filenames = None
        self._train_batch_generator = None
        self._val_batch_generator = None
        if os.path.exists(self._data_augment_merged_dir):
            self._set_generator()
        if model_keras:
            self.model_path = '{}.h5'.format(self._model_name)
        self._model = None
        self._compile_params = compile_params
        
        self._model_prefix = os.path.join(
            self._data_augment_dir_prefix, self._model_name)
        self._weights_dir = os.path.join('weights', self._model_prefix)
        self._result_dir = os.path.join('result', self._model_prefix)
        self._predict_dir = os.path.join('predict', self._model_prefix)
        
        self._num_training_samples = len(self._train_filenames)*self._original_batch_size
        self._num_validation_samples = len(self._val_filenames)*self._original_batch_size
        
        self._monitor = monitor
        self._monitor_mode = monitor_mode
        self._filepath = None
        self._checkpoint = None
        self._csv_logger = None
        self._callbacks_list = None
        
        self.history = None
        if training:
            self._set_checkpoint()

        # Strategy objects
        self._preprocess_strategy_context = PreprocessStrategyContext(
            self._preprocessed_type)
        
        
    def _get_data_files(self):
        data_types = ['train', 'val', 'test']
        file_types = ['data', 'target', 'mask']
        outputs = {}
        for data_type in data_types:
            outputs[data_type] = {}
            for file_type in file_types:
                outputs[data_type][file_type] = get_data_file_path(
                    self._preprocessed_data_dir, self._used_band, 
                    self._input_timesteps, self._output_timesteps,
                    data_type, file_type)
        return outputs
                

    def create_water_cloud_mask(self):
        create_water_cloud_mask(
            self._raw_data_dir, self._used_band, self._year_range,
            self._n_data_per_year, self._day_period, self._mask_data_dir)

    
    def preprocess_data(self):
        change_fill_value_data_dir = os.path.join(
            self._preprocessed_data_dir_prefix, 'change_fill_value')
        change_fill_value(
            self._raw_data_dir, self._used_band, self._year_range,
            self._n_data_per_year, self._day_period, change_fill_value_data_dir)
        self._preprocess_strategy_context.preprocess_data(
            change_fill_value_data_dir, '', self._year_range,
            self._n_data_per_year, self._day_period, self._preprocessed_data_dir)
        
    
    def make_archive_masked_data(self):
        make_archive('masked_data', 'zip', '.', self._mask_data_dir)
        
    def make_archive_preprocessed_data(self):
        make_archive('preprocessed_data', 'zip', '.', self._preprocessed_data_dir)
    
    
    def create_data_file(self):
        outputs = create_data_file_continuous_years(
            self._preprocessed_data_dir, self._input_timesteps,
            self._output_timesteps, self._list_years_train, self._list_years_val,
            self._list_years_test, self._mask_data_dir)
        make_archive('data_file', 'zip', '.', 'data_file')
        return outputs
    
    
    def augment_data(self, n_samples=50):
        for data_type in ['train', 'val']:
            data_augment_dir = os.path.join(self._data_augment_dir, data_type)
            augment_one_reservoir_without_cache(
                self._data_files, data_augment_dir, 
                self._crop_size, n_samples, data_type,
                self._input_timesteps, self._output_timesteps)
            
            data_augment_merged_dir = os.path.join(self._data_augment_merged_dir, data_type)
            merge_data_augment(data_augment_dir, data_augment_merged_dir)
        
    def make_archive_augment_data(self):
        make_archive('data_augment_merged', 'zip', '.', self._data_augment_merged_dir)
    
    
    def _set_generator(self):        
        train_dir = os.path.join(self._data_augment_merged_dir, 'train')
        self._train_filenames = [os.path.join(train_dir, data_index)
                                 for data_index in os.listdir(train_dir)]
        self._train_batch_generator = get_generator(
            self._model_name, self._train_filenames, self._batch_size,
            original_batch_size=self._original_batch_size)
        
        val_dir = os.path.join(self._data_augment_merged_dir, 'val')
        self._val_filenames = [os.path.join(val_dir, data_index)
                               for data_index in os.listdir(val_dir)]
        self._val_batch_generator = get_generator(
            self._model_name, self._val_filenames, self._batch_size,
            original_batch_size=self._original_batch_size)
        
    def get_train_generator(self):
        if self._train_batch_generator is None:
            self._set_generator
        return self._train_batch_generator
    
    def get_val_generator(self):
        if self._train_batch_generator is None:
            self._set_generator
        return self._val_batch_generator
    
    def create_model(self):
        K.clear_session()
        self._model = create_model_by_name(
            self._model_name, self._crop_size, self._input_timesteps,
            self._output_timesteps, self._compile_params)
        return self._model
    
    def plot_model(self):
        if self._model is not None:
            plot_model(self._model, to_file='{}.png'.format(
                self._model_name), show_shapes=True)
            model_plot = misc.imread('{}.png'.format(self._model_name))
            plt.figure(figsize=(15,15))
            plt.imshow(model_plot)
            plt.show()
            
    def summary(self):
        if self._model is not None:
            self._model.summary()
            
    def _set_checkpoint(self):
        if not os.path.exists(self._weights_dir):
            os.makedirs(self._weights_dir)
        self._filepath = os.path.join(self._weights_dir, "weights-{epoch:02d}.h5")
        self._checkpoint = ModelCheckpoint(
            self._filepath, monitor=self._monitor, mode=self._monitor_mode, verbose=1)
        self._csv_logger = CSVLogger(os.path.join(
            self._weights_dir, 'log.csv'), append=True, separator=';')
        self._callbacks_list = [self._checkpoint, self._csv_logger]
            

    def train(self, epochs=50):
        self.history = self._model.fit_generator(
            generator=self._train_batch_generator,
            steps_per_epoch=(self._num_training_samples // self._batch_size),
            epochs=epochs,
            validation_data=self._val_batch_generator,
            validation_steps=(self._num_validation_samples // self._batch_size),
            callbacks=self._callbacks_list)
        self._model.save(self.model_path)
    
    def load_model(self):
        if os.path.exists(self.model_path):
            model = self.create_model()
            model.load(self.model_path)
            self._model = compile_model(model, self._compile_params)
            return self._model
        return None
    
    def inference(self, model=None, data_type='test', idx=0):
        if model is None:
            model = self._model
        assert model is not None
        file_path = self._data_files[data_type]['data']
        data_test = get_data_test(file_path, idx)
        data_test = np.expand_dims(np.expand_dims(data_test, axis=0), axis=-1)
        output = model.predict(data_test)
        cache_data(output, os.path.join(self._predict_dir, '{}.dat'.format(idx)))
        return output
    
    def inference_all(self, model=None, data_type='test'):
        if model is None:
            model = self._model
        assert model is not None
        file_path = self._data_files[data_type]['data']
        n = len(get_data_paths(data_file_path))
        for idx in n:
            self.inference(model, data_type, idx)
    
    def model_eval(self, data_type='test', idx=0):
        pass
    
    def model_eval_all(self, data_type='test'):
        pass

    def visualize_result(self, data_type='test', idx=0, save_result=True):
        pass
    
    def get_predict_water_area(self, data_type='test'):
        pass
