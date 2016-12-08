"""
Some common constants and utilities for this project.
"""

import os

IMAGE_COLUMNS = ['center_image', 'left_image', 'right_image']
CONTROL_COLUMNS = ['steering_angle', 'throttle', 'brake']
TELEMETRY_COLUMNS = ['speed']

IMAGE_SHAPE = (160, 320, 3)
DRIVING_LOG_CSV = 'driving_log.csv'
DRIVING_LOG_PICKLE = 'driving_log.p'
BOTTLENECK_PICKLE = 'bottleneck.p'

BASE_MODEL_JSON_FILE = 'base_model.json'
BASE_MODEL_WEIGHTS_FILE = 'base_model.h5'

def get_bottleneck_folder(data_dir):
    return os.path.join(data_dir, 'bottleneck')

def get_bottleneck_pathname(data_dir, index):
    return os.path.join(get_bottleneck_folder(data_dir), '%04d.npz' % index)

