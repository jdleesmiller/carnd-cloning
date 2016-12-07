import os

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.misc import imread
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.models import Model, Sequential

from common import *

DATA_DIR = 'data/clean_1'

DRIVING_LOG = pd.read_pickle(os.path.join(DATA_DIR, DRIVING_LOG_PICKLE))

BOTTLENECK_FEATURES = np.load(os.path.join(DATA_DIR,
    get_bottleneck_npy_filename('center_image')))

def split_training_set():
    return train_test_split(
        np.arange(len(DRIVING_LOG)),
        DRIVING_LOG['smooth_steering_angle_1'].values,
        test_size=0.2,
        random_state=42)

def run():
    x_train_index, x_val_index, y_train, y_val = split_training_set()
    x_train = BOTTLENECK_FEATURES[x_train_index]
    x_val = BOTTLENECK_FEATURES[x_val_index]

    model = Sequential()
    model.add(Flatten(input_shape=x_train[0].shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    model.fit(x_train, y_train,
        nb_epoch=2,
        batch_size=32,
        validation_data=(x_val, y_val))

if __name__ == '__main__':
    run()

