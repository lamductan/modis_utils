from tensorflow.python.keras.utils import Sequence
from modis_utils.misc import restore_data

def get_generator(model_name, data_filenames, batch_size, original_batch_size):
    if model_name == 'convlstm_simple':
        return OneOutputGenerator(data_filenames, batch_size, original_batch_size)
    elif model_name == 'convlstm_reconstruct':
        return MultipleOutputGenerator(data_filenames, batch_size, original_batch_size)


class MyGenerator(Sequence):
    def __init__(self, data_filenames, batch_size, original_batch_size):
        assert batch_size < original_batch_size
        
        self.data_filenames = data_filenames
        self.batch_size = batch_size
        self.original_batch_size = original_batch_size
        self.k = self.original_batch_size // self.batch_size

    def __len__(self):
        return len(self.data_filenames)*self.k


class OneOutputGenerator(MyGenerator):
      
    def __getitem__(self, idx):         
        data = restore_data(self.data_filenames[idx // self.k])
        i = idx % self.k
        batch_X = data[0][i*self.batch_size:(i+1)*self.batch_size]
        batch_y = data[1][i*self.batch_size:(i+1)*self.batch_size]
        return batch_X, batch_y


class MultipleOutputGenerator(MyGenerator):
      
    def __getitem__(self, idx):         
        data = restore_data(self.data_filenames[idx // self.k])
        i = idx % self.k
        batch_X = data[0][i*self.batch_size:(i+1)*self.batch_size]
        batch_y = data[1][i*self.batch_size:(i+1)*self.batch_size]
        return batch_X, [batch_y, batch_X]