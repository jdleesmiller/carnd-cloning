import os

import numpy as np
import pandas as pd

from common import *

def load_driving_log(path):
    """
    Read in the driving log CSV and do some basic transforms.
    """
    log = pd.read_csv(
        path,
        header=None,
        names=IMAGE_COLUMNS + CONTROL_COLUMNS + TELEMETRY_COLUMNS)

    # Get rid of the original image paths. (I've moved the files.)
    log[IMAGE_COLUMNS] = log[IMAGE_COLUMNS].applymap(os.path.basename)

    # Find delta t between frames from the image path names for smoothing.
    log['time'] = pd.to_datetime(
        log['center_image'], format='center_%Y_%m_%d_%H_%M_%S_%f.jpg')

    # Add the correct paths, based on the location of the CSV file.
    path_root = os.path.dirname(path)
    log[IMAGE_COLUMNS] = log[IMAGE_COLUMNS].applymap(
        lambda basename: os.path.join(path_root, 'IMG', basename))

    return log

def smooth(values, dt, tau):
    """
    Apply smoothing for an unevenly spaced timeseries. Formula is from
    http://www.eckner.com/papers/ts_alg.pdf
    """
    result = np.empty(len(values))
    result[0] = values[0]
    weights = np.exp(-dt / tau)
    for i in range(1, len(values)):
        result[i] = weights[i] * result[i - 1] + (1 - weights[i]) * values[i]

    return result

def smooth_control_inputs(log, tau):
    """
    Bind smoothed control inputs to the driving log. This uses an exponential
    moving average with time constant tau, and it averages both a forward and
    a backward average. The weight for a measurement is $1 - exp(dt / tau)$,
    where dt is the time since the last measurement.
    """
    dt_prev =  log['time'].diff( 1).map(lambda t: t.total_seconds())
    dt_next = -log['time'].diff(-1).map(lambda t: t.total_seconds())
    for control_column in CONTROL_COLUMNS:
        smooth_forward = smooth(log[control_column], dt_prev, tau)
        smooth_backward = smooth(
            np.array(log[control_column])[::-1],
            np.array(dt_next)[::-1],
            tau)[::-1]
        smooth_stack = np.vstack((smooth_forward, smooth_backward))
        column_name = 'smooth_%s_%g' % (control_column, tau)
        log[column_name] = np.mean(smooth_stack, 0)
    return log

