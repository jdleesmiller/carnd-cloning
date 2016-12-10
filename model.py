import os

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.layers import Input, Flatten, Dense, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.regularizers import l2

from scipy.misc import imread
from sklearn.model_selection import train_test_split

from common import *

def load_bottleneck_files(log):
    return log['bottleneck_features'].map(
        lambda pathname: np.load(pathname)
    ).values

def load_bottleneck_features(files, column):
    return np.array([files[index][column] for index in range(len(files))])

def make_side_camera_data(files, side_camera_bias,
        center_features, center_labels):

    left_features = load_bottleneck_features(files, 'left_image')
    left_labels = center_labels + side_camera_bias

    right_features = load_bottleneck_features(files, 'right_image')
    right_labels = center_labels - side_camera_bias

    features = np.concatenate([left_features, center_features, right_features])
    labels = np.concatenate([left_labels, center_labels, right_labels])

    return features, labels

def generate_data(log,
        label_column='smooth_steering_angle_1',
        side_camera_bias=None,
        batch_size=32):
    start = 0
    while True:
        end = start + batch_size
        batch = log.iloc[start:end]

        batch_files = load_bottleneck_files(batch)

        center_features = load_bottleneck_features(batch_files, 'center_image')
        center_labels = batch[label_column].values

        if side_camera_bias is None:
            yield center_features, center_labels
        else:
            yield make_side_camera_data(
                batch_files, side_camera_bias, center_features, center_labels)

        start = end
        if start >= len(log):
            start = 0

def split_training_set(log, test_size=0.2, random_state=42):
    return train_test_split(
        np.arange(len(log)),
        test_size=test_size,
        random_state=random_state)

def build(input_shape, nb_filter=64, l2_weight=0.01):
    np.random.seed(42)

    model = Sequential()
    model.add(Convolution2D(nb_filter=nb_filter, nb_row=1, nb_col=1,
        input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1, W_regularizer=l2(l2_weight)))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    return model

def train(model, log,
        label_column='smooth_steering_angle_1',
        side_camera_bias=None,
        test_size=0.2,
        nb_epoch=1,
        batch_size=32):

    x_train_indexes, x_val_indexes = \
        split_training_set(log, test_size=test_size)

    log_train = log.iloc[x_train_indexes]
    log_val = log.iloc[x_val_indexes]

    callbacks = [EarlyStopping(patience=2)]
    training_generator = \
        generate_data(log_train, label_column=label_column,
            batch_size=batch_size, side_camera_bias=side_camera_bias)
    validation_generator = \
        generate_data(log_val, label_column=label_column,
            batch_size=batch_size, side_camera_bias=side_camera_bias)

    samples_per_epoch = len(x_train_indexes)
    nb_val_samples = len(x_val_indexes)
    if side_camera_bias is not None:
        samples_per_epoch *= 3
        nb_val_samples *= 3

    history = model.fit_generator(
        training_generator,
        samples_per_epoch=samples_per_epoch,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        callbacks=callbacks)

    return history

if __name__ == '__main__':
    pass
