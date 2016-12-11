import os

import numpy as np

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

def make_batch(batch, label_column, side_camera_bias):
    batch_files = load_bottleneck_files(batch)

    features = load_bottleneck_features(batch_files, 'center_image')
    labels = batch[label_column].values
    if side_camera_bias is not None:
        features, labels = make_side_camera_data(
            batch_files, side_camera_bias, features, labels)

    for batch_file in batch_files:
        batch_file.close()

    return features, labels

def make_batches(folder, log, batch_size, label_column, side_camera_bias):
    """
    Precompute batches. This means we can't shuffle between epochs,
    but it's much faster to load one batch with one operation than to load
    individual files. (We do still shuffle at the start.)
    """
    batch_pathnames = []
    nb_samples = 0
    for start in range(0, len(log), batch_size):
        end = start + batch_size
        batch = log.iloc[start:end]
        nb_samples += len(batch)

        features, labels = make_features(batch, label_column, side_camera_bias)

        batch_pathname = os.path.join(folder, '%04d.npz' % start)
        np.savez(batch_pathname, features=features, labels=labels)
        batch_pathnames.append(batch_pathname)

    return nb_samples, batch_pathnames

def split_training_set(log, test_size, random_state):
    return train_test_split(
        np.arange(len(log)),
        test_size=test_size,
        random_state=random_state)

def load_existing_batches(folder):
    pathnames = [
        os.path.join(folder, filename) for filename in os.listdir(folder)
    ]

    nb_samples = 0
    for pathname in pathnames:
        with np.load(pathname) as batch:
            nb_samples += len(batch['labels'])

    return nb_samples, pathnames

def make_train_val_batches(log, key):
    folder = os.path.join('batches', make_filestem('batch', key))
    folder_train = os.path.join(folder, 'train')
    folder_val = os.path.join(folder, 'val')

    if os.path.isdir(folder_train) and os.path.isdir(folder_val):
        print(folder, 'for batches exists')
        nb_train, batches_train = load_existing_batches(folder_train)
        nb_val, batches_val = load_existing_batches(folder_val)

        if key['side_camera_bias'] is None:
            assert nb_val + nb_train == len(log)
        else:
            assert nb_val + nb_train == 3 * len(log)

        return nb_train, batches_train, nb_val, batches_val

    os.makedirs(folder_train, exist_ok=True)
    os.makedirs(folder_val, exist_ok=True)

    x_train_indexes, x_val_indexes = \
        split_training_set(log,
            test_size=key['test_size'],
            random_state=key['random_state'])

    log_train = log.iloc[x_train_indexes]
    log_val = log.iloc[x_val_indexes]

    nb_train, batches_train = make_batches(folder_train, log_train,
        key['batch_size'], key['label_column'], key['side_camera_bias'])
    nb_val, batches_val = make_batches(folder_val, log_val,
        key['batch_size'], key['label_column'], key['side_camera_bias'])

    return nb_train, batches_train, nb_val, batches_val

