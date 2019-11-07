import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras import optimizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class Config:
    def __init__(self):
        # data set
        self.classes = {'dog': 0, 'cat':1}
        self.data_set = 'data_set/train'

        # image
        self.width = 256
        self.height = 256
        # training parameters
        self.predict_num = 2
        self.batch_size = 32 
        self.epochs = 10
        self.optimizers = optimizers.Adam
        self.learning_rate = 0.0001
        #self.loss = 'mean_squared_error'
        self.loss = 'categorical_crossentropy'
        self.checkpoints_save_path = 'checkpoints/model_best.h5'
        self.early_stopping = EarlyStopping(patience=100)
        self.model_save_path = 'checkpoints/model_best_final.h5'
        # set GPU        
        self.gpus = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        set_session(tf.Session(config=cfg))

        
    def data_load(self):
        path = os.path.join(self.data_set)
        name_list = os.listdir(path)
        num = len(name_list)
        X = np.zeros((num, self.width, self.height, 3), dtype=np.uint8)
        y = np.zeros((num, self.predict_num), dtype=np.uint8)
        for i in range(num):
            name = name_list[i].split('.')[0]
            img_name = os.path.join(path, name_list[i])
            X[i] = cv2.resize(cv2.imread(img_name), (self.width, self.height))
            if name == 'dog':
                y[i, 0] = 1
            else:
                y[i, 1] = 1
        #print(y)
        return X, y



