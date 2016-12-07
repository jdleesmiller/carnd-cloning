"""
Some common constants and utilities for this project.
"""

IMAGE_COLUMNS = ['center_image', 'left_image', 'right_image']
CONTROL_COLUMNS = ['steering_angle', 'throttle', 'brake']
TELEMETRY_COLUMNS = ['speed']

IMAGE_SHAPE = (160, 320, 3)
DRIVING_LOG_CSV = 'driving_log.csv'
DRIVING_LOG_PICKLE = 'driving_log.p'
BOTTLENECK_PICKLE = 'bottleneck.p'

def get_bottleneck_npy_filename(image_column):
    return 'bottleneck_%s.npy' % image_column
