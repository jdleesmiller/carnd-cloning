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

def generate_data(data_dir, log, indexes, batch_size=32):
    start = 0
    while True:
        end = start + batch_size
        batch_indexes = indexes[start:end]

        batch_features = np.array([
            np.load(get_bottleneck_filename(data_dir, index))['center_image']
            for index in batch_indexes
        ])

        batch_labels = log['smooth_steering_angle_1'].values[batch_indexes]

        start = end
        if start >= len(indexes):
            start = 0

        yield(batch_features, batch_labels)

def split_training_set(log, test_size=0.2, random_state=42):
    return train_test_split(
        np.arange(len(log)),
        test_size=test_size,
        random_state=random_state)

def build(input_shape):
    np.random.seed(42)

    model = Sequential()
    model.add(Convolution2D(nb_filter=64, nb_row=1, nb_col=1,
        input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1, W_regularizer=l2(0.01)))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    return model

def train(model, data_dir, log,
        test_size=0.2,
        nb_epoch=1,
        batch_size=32):

    x_train_indexes, x_val_indexes = \
        split_training_set(log, test_size=test_size)

    callbacks = [EarlyStopping(patience=2)]
    training_generator = \
        generate_data(data_dir, log, x_train_indexes, batch_size=batch_size)
    validation_generator = \
        generate_data(data_dir, log, x_val_indexes, batch_size=batch_size)

    model.fit_generator(
        training_generator,
        samples_per_epoch=len(x_train_indexes),
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=len(val_indexes),
        callbacks=callbacks)

    return model

if __name__ == '__main__':
    bottleneck_features = np.load(os.path.join(DATA_DIR,
        get_bottleneck_filename('center_image')))['arr_0']
    train(bottleneck_features)

