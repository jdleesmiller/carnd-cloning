"""
Generate bottleneck features by running them through the inception network.
"""
import os
import pickle

import numpy as np
import pandas as pd
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Input
from scipy.misc import imread

from common import *
import model_io

def generate_data(filenames, batch_size=32):
    start = 0
    end = start + batch_size
    n = filenames.shape[0]

    while True:
        print('batch:', start, 'to', end, 'of', n)
        filenames_batch = filenames[start:end]

        X_batch = [imread(filename) for filename in filenames_batch]
        X_batch = np.array(X_batch).astype(np.float32)
        X_batch = preprocess_input(X_batch)

        y_batch = np.zeros(end - start) # We need to pass it, but it is ignored.

        start += batch_size
        end += batch_size
        if start >= n:
            start = 0
            end = batch_size

        yield (X_batch, y_batch)

def bottleneck(filenames):
    base_model = model_io.load_base_model()
    return base_model.predict_generator(
        generate_data(filenames), filenames.shape[0])

def run(data_dir):
    input_pathname = os.path.join(data_dir, DRIVING_LOG_PICKLE)
    driving_log = pd.read_pickle(input_pathname)
    driving_log = driving_log[:3]

    for image_column in IMAGE_COLUMNS[1:2]:
        print('column:', image_column)
        output_pathname = os.path.join(
                data_dir, get_bottleneck_filename(image_column))

        result = bottleneck(driving_log[image_column])
        np.savez_compressed(output_pathname, result)

if __name__ == '__main__':
    run()

