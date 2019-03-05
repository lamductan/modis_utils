import os
from modis_utils.misc import restore_data, cache_data


class PreprocessStrategy:
    def __init__(self):
        self.fn = None

    def preprocess_data(self, data_dir, used_band, year_range,
                        n_data_per_year, day_period, preprocessed_data_dir):
        for year in range(year_range[0], year_range[1] + 1):
            for d in range(n_data_per_year):
                day = d*day_period + 1
                prefix = os.path.join(str(year), str(year) + str(day).zfill(3))
                current_data_dir = os.path.join(data_dir, prefix)
                try:
                    list_imgs = os.listdir(current_data_dir)
                    filename = list(filter(lambda x: used_band in x, list_imgs))[0]
                    img = restore_data(os.path.join(current_data_dir, filename))
                    normalized_img = self.fn(img)
                    cur_dest_dir = os.path.join(preprocessed_data_dir, prefix)
                    try:
                        os.makedirs(cur_dest_dir)
                    except:
                        pass
                    cache_data(normalized_img, os.path.join(cur_dest_dir, 
                                                            'preprocessed.dat'))
                except:
                    print('Not found data {}{:03} in {}.'.format(
                        year, day, current_data_dir))


class NormalizedDivStrategy(PreprocessStrategy):
    def __init__(self):
        self.fn = lambda x: x/10000
        