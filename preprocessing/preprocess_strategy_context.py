from modis_utils.preprocessing.preprocess_strategy import NormalizedDivStrategy

class PreprocessStrategyContext:

    def __init__(self, preprocessed_type):
        self.strategy = None
        if preprocessed_type == 'normalized_div':
            self.strategy = NormalizedDivStrategy()
        else:
            raise ValueError

    def preprocess_data(self, data_dir, used_band, year_range,
                        n_data_per_year, day_period, preprocessed_data_dir):
        self.strategy.preprocess_data(data_dir, used_band, year_range,
                                      n_data_per_year, day_period, preprocessed_data_dir)

    def inverse(self, data):
        return self.strategy.inverse(data)
