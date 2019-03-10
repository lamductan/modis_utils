from modis_utils.model.convlstm_simple import ConvLSTMSimpleOneTimeStepsOutput
from modis_utils.model.convlstm_simple import ConvLSTMSimpleSequenceTimeStepsOutput
from modis_utils.model.cplx_model import SkipConvLSTMSingleOutput
from modis_utils.model.resnet_lstm import ResnetLSTMSingleOutput


def get_model_utils(model_name, output_timesteps):
    if model_name == 'convlstm_simple':
        if output_timesteps == 1:
            return ConvLSTMSimpleOneTimeStepsOutput
        else:
            return ConvLSTMSimpleSequenceTimeStepsOutput
    elif model_name == 'skip_conv_single_output':
        return SkipConvLSTMSingleOutput
    elif model_name == 'resnet_lstm':
        return ResnetLSTMSingleOutput