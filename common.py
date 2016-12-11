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

def base_model_stem(cut_index):
    return 'base_model_%d' % cut_index

def make_filestem(prefix, params):
    stem = prefix
    for param in sorted(params):
        value = params[param]
        if value is None: value = 'None'
        stem += '.' + param + '-' + str(value)
    return stem

