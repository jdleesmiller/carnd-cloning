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

def make_features_and_labels_for_center_cam(
        log, bottleneck_files, batch_indexes, label_column):
    batch_features = np.array([
        bottleneck_files[index]['center_image']
        for index in batch_indexes
    ])
    batch_labels = log[label_column].values[batch_indexes]
    return batch_features, batch_labels

def make_features_and_labels_for_all_cams(
        log, bottleneck_files, batch_indexes, label_column,
        side_camera_bias):

    center_features, center_labels = make_features_and_labels_for_center_cam(
        log, bottleneck_files, batch_indexes)

    left_features = np.array([
        bottleneck_files[index]['left_image']
        for index in batch_indexes
    ])
    left_labels = center_labels + side_camera_bias

    right_features =  np.array([
        bottleneck_files[index]['right_image']
        for index in batch_indexes
    ])
    right_labels = center_labels - side_camera_bias

    features = np.concatenate([left_features, center_features, right_features])
    labels = np.concatenate([left_labels, center_labels, right_labels])

    return features, labels

def generate_data(log, indexes,
        label_column='smooth_steering_angle_1',
        side_camera_bias=None,
        batch_size=32):
    # I think this uses weakrefs, so we can cache the files here.
    bottleneck_files = log['bottleneck_features'].map(
        lambda pathname: np.load(pathname)
    ).values
    start = 0
    while True:
        end = start + batch_size
        batch_indexes = indexes[start:end]

        if side_camera_bias is None:
            yield make_features_and_labels_for_center_cam(
                log, bottleneck_files, batch_indexes, label_column)
        else:
            yield make_features_and_labels_for_all_cams(
                log, bottleneck_files, batch_indexes, label_column,
                side_camera_bias)

        start = end
        if start >= len(indexes):
            start = 0

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

def train(model, log,
        test_size=0.2,
        nb_epoch=1,
        batch_size=32):

    x_train_indexes, x_val_indexes = \
        split_training_set(log, test_size=test_size)

    callbacks = [EarlyStopping(patience=1)]
    training_generator = \
        generate_data(log, x_train_indexes, batch_size=batch_size)
    validation_generator = \
        generate_data(log, x_val_indexes, batch_size=batch_size)

    model.fit_generator(
        training_generator,
        samples_per_epoch=len(x_train_indexes),
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=len(x_val_indexes),
        callbacks=callbacks)

    return model

if __name__ == '__main__':
    pass
