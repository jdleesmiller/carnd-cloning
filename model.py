import os

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.regularizers import l2
import keras.initializations

from common import *

def small_normal(shape, name=None):
    return keras.initializations.normal(shape, scale=0.1, name=name)

# So we can use model_from_json.
setattr(keras.initializations, 'small_normal', small_normal)

def build(input_shape, nb_filter, nb_hidden, l2_weight, optimizer):
    model = Sequential()
    model.add(Convolution2D(nb_filter=nb_filter, nb_row=1, nb_col=1,
        input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(nb_hidden, activation='tanh', init=small_normal, W_regularizer=l2(l2_weight)))
    model.add(Dense(1, init=small_normal, W_regularizer=l2(l2_weight)))

    model.compile(optimizer=optimizer, loss='mean_absolute_error')
    model.summary()

    return model

def generate_data(batch_files):
    while True:
        for batch_file in batch_files:
            with np.load(batch_file) as batch:
                yield batch['features'], batch['labels']

def train(model, nb_epoch, patience,
        nb_train, batch_files_train, nb_val, batch_files_val,
        save_stem=None):

    callbacks = [EarlyStopping(patience=patience)]
    if save_stem is not None:
        with open(save_stem + '.json', 'w') as model_file:
            model_file.write(model.to_json())
        callbacks.append(ModelCheckpoint(
            save_stem + '.h5', save_best_only=True, save_weights_only=True))

    training_generator = generate_data(batch_files_train)
    validation_generator = generate_data(batch_files_val)

    history = model.fit_generator(
        training_generator,
        samples_per_epoch=nb_train,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val,
        callbacks=callbacks)

    return history

if __name__ == '__main__':
    pass
