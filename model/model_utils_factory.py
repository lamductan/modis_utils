from modis_utils.model.convlstm_simple import ConvLSTMSimpleOneTimeStepsOutput
from modis_utils.model.convlstm_simple import ConvLSTMSimpleSequenceTimeStepsOutput


def get_model_utils(model_name, output_timesteps):
    if model_name == 'convlstm_simple':
        if output_timesteps == 1:
            return ConvLSTMSimpleOneTimeStepsOutput
        else:
            return ConvLSTMSimpleSequenceTimeStepsOutput