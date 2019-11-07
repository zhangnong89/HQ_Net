import config as config
import model.network as network
from keras.utils.training_utils import multi_gpu_model
from keras.backend import image_data_format
from keras.layers import Dense, GlobalAveragePooling2D, Activation, Lambda
from keras.models import Input, Model
from keras.utils import plot_model


class NN_Model:
    def __init__(self):
        self.cfg = config.Config()
    
    def model(self):
        # build the network
        if image_data_format() == 'channels_last':
            input_shape = (None, None, 3)
        else:
            input_shape = (3, None, None)
        inputs = Input(shape=input_shape)
        predictions = network.nn_base(inputs)
        single_model = Model(inputs=inputs, outputs=predictions)
        #plot_model(single_model, to_file='model.png')
        single_model.summary()
        num_gpus = int(len(self.cfg.gpus.split(",")))
        if num_gpus == 1:
             return single_model
        else:
             parallel_model = multi_gpu_model(single_model, gpus=num_gpus)
             return parallel_model, single_model
