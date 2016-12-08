"""
Generate bottleneck features by running them through the inception network.
"""
import os

import numpy as np
from keras.applications.inception_v3 import preprocess_input
from scipy.misc import imread

from common import *
import model_io

# load a batch of rows
# run predictions for each of the cameras
# save an npz file for each frame with the bottleneck results

def chunks(itr, size):
    for i in range(0, len(itr), size):
        yield itr[i:(i+size)]

def run(log, data_dir, batch_size=32):
    base_model = model_io.load_base_model()
    os.makedirs(get_bottleneck_folder(data_dir), exist_ok=True)

    index = 0
    for batch in chunks(log, batch_size):
        # These are in center_0, left_0, right_0, center_1, left_1, ... order.
        filenames = [
            batch[column].values[i]
            for i in range(len(batch))
            for column in IMAGE_COLUMNS
        ]

        X_batch = [imread(filename) for filename in filenames]
        X_batch = np.array(X_batch).astype(np.float32)
        X_batch = preprocess_input(X_batch)

        X_base = base_model.predict(X_batch)

        for prediction in chunks(X_base, len(IMAGE_COLUMNS)):
            if index % 50 == 0:
                print('index', index)
            np.savez(get_bottleneck_pathname(data_dir, index), **{
                IMAGE_COLUMNS[i]: prediction[i] for i in range(len(prediction))
            })
            index += 1

if __name__ == '__main__':
    run()

