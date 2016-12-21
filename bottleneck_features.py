"""
Generate bottleneck features by running them through the inception network.
"""
import os

import numpy as np
from keras.applications.inception_v3 import preprocess_input
from scipy.misc import imread

from common import *
import model_io

def get_bottleneck_folder(data_dir, cut_index):
    return os.path.join(data_dir, 'bottleneck_%d' % cut_index)

def get_bottleneck_pathname(data_dir, cut_index, index):
    return os.path.join(
        get_bottleneck_folder(data_dir, cut_index), '%04d.npz' % index)

def chunks(itr, size):
    for i in range(0, len(itr), size):
        yield itr[i:(i+size)]

def bind_bottleneck_features(log, data_dir, cut_index):
    """
    If we have lost the bottleneck feature mapping for a log, just rebuild it
    rather than re-bottlenecking all the data.
    """
    log['bottleneck_features'] = [
        get_bottleneck_pathname(data_dir, cut_index, index)
        for index in range(len(log))
    ]
    return log

def make_batch_images(log):
    # These are in center_0, left_0, right_0, center_1, left_1, ... order.
    filenames = [
        log[column].values[i]
        for i in range(len(log))
        for column in IMAGE_COLUMNS
    ]

    # These are in center_0, center_0_flipped, left_0, left_0_flipped, ...
    # order.
    images = [imread(filename) for filename in filenames]
    images = [[image, np.fliplr(image)] for image in images]
    images = [im for ims in images for im in ims]

    column_names = [[n, 'flipped_' + n] for n in IMAGE_COLUMNS]
    column_names = [n for ns in column_names for n in ns]

    return images, column_names

def run(log, data_dir, cut_index, batch_size=32):
    bottleneck_folder = get_bottleneck_folder(data_dir, cut_index)
    if os.path.isdir(bottleneck_folder):
        print(bottleneck_folder, 'exists; just binding.')
        return bind_bottleneck_features(log, data_dir, cut_index)

    os.makedirs(bottleneck_folder, exist_ok=True)

    base_model = model_io.load_base_model(cut_index)

    index = 0
    log['bottleneck_features'] = ''
    for batch in chunks(log, batch_size):
        images, column_names = make_batch_images(batch)
        X_batch = np.array(images).astype(np.float32)
        X_batch = preprocess_input(X_batch)
        X_base = base_model.predict(X_batch)

        for prediction in chunks(X_base, len(column_names)):
            if index % 50 == 0:
                print('index', index)
            output_pathname = get_bottleneck_pathname(
                data_dir, cut_index, index)
            np.savez(output_pathname, **{
                column_names[i]: prediction[i]
                for i in range(len(prediction))
            })
            log['bottleneck_features'].values[index] = output_pathname
            index += 1

    return log

