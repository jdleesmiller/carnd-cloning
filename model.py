import os

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.misc import imread

from keras.callbacks import EarlyStopping
from keras.layers import Input, Flatten, Dense, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.regularizers import l2

from common import *

# If we have generators for train and val, we'll probably want to do this.
# def split_training_set():
#     return train_test_split(
#         np.arange(len(DRIVING_LOG)),
#         DRIVING_LOG['smooth_steering_angle_1'].values,
#         test_size=0.2,
#         random_state=42)

def train(features, labels, nb_epoch=1, batch_size=32, validation_split=0.2):
    np.random.seed(42)

    model = Sequential()
    model.add(Convolution2D(nb_filter=64, nb_row=1, nb_col=1, input_shape=features[0].shape))
    model.add(Flatten())
    model.add(Dense(1, W_regularizer=l2(0.01)))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    callbacks = [EarlyStopping(patience=2)]

    model.fit(features, labels,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks = callbacks)

    return model

if __name__ == '__main__':
    bottleneck_features = np.load(os.path.join(DATA_DIR,
        get_bottleneck_filename('center_image')))['arr_0']
    train(bottleneck_features)

