import os

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.misc import imread
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.models import Model, Sequential

# no bottleneck; no generator; just get it working
# can test on training data for now, but of course we will want to do something
# to make a validation set

origin_height = 160
origin_width = 320

DATA_DIR = 'data/clean_1'
DRIVING_LOG_PICKLE = 'driving_log.p'
IMAGE_SHAPE = (160, 320, 3)

DRIVING_LOG = pd.read_pickle(os.path.join(DATA_DIR, DRIVING_LOG_PICKLE))

def split_training_set():
    return train_test_split(
        DRIVING_LOG['center_image'].values,
        DRIVING_LOG['smooth_steering_angle_1'].values,
        test_size=0.2,
        random_state=42)

def generate_data(file_names, labels, batch_size=16):
    start = 0
    end = start + batch_size
    n = file_names.shape[0]

    while True:
        file_names_batch = file_names[start:end]
        X_batch = [
            imread(os.path.join(DATA_DIR, 'IMG', basename)) \
                for basename in file_names_batch
        ]
        X_batch = np.array(X_batch).astype(np.float32)
        X_batch = preprocess_input(X_batch)

        y_batch = labels[start:end]

        start += batch_size
        end += batch_size
        if start >= n:
            start = 0
            end = batch_size

        yield (X_batch, y_batch)

def run():
    X_file_train, X_file_val, y_train, y_val = split_training_set()

    print('building base model')
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=Input(shape=IMAGE_SHAPE))

    print('building custom model')
    output = base_model.output
    # TODO: doesn't work: output = tf.stop_gradient(output)
    output = Flatten()(output)
    output = Dense(1)(output)

    print('compiling')
    model = Model(input=base_model.input, output=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    print('fitting')
    model.fit_generator(
        generator=generate_data(X_file_train, y_train),
        samples_per_epoch=32,
        nb_epoch=2,
        validation_data=generate_data(X_file_val, y_val),
        nb_val_samples=32)

if __name__ == '__main__':
    run()

