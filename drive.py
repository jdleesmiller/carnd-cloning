import argparse
import base64

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from keras.applications.inception_v3 import preprocess_input

from common import *
import model
import model_io

sio = socketio.Server()
app = Flask(__name__)
top_model = None
prev_image_array = None
weights_file = None
weights_file_mtime = None

base_model = model_io.load_base_model(12)

def check_for_weights_update():
    global weights_file_mtime
    """
    If the weights file changes, reload it without having to tear down the
    server.
    """
    latest_mtime = os.stat(weights_file).st_mtime
    if weights_file_mtime is None:
        weights_file_mtime = latest_mtime
        return

    if weights_file_mtime == latest_mtime:
        return

    print('reloading weights')
    model.load_weights(weights_file)
    weights_file_mtime = latest_mtime

@sio.on('telemetry')
def telemetry(sid, data):
    start_time = time.time()

    # The current steering angle of the car (in degrees)
    steering_angle = float(data["steering_angle"]) / 25.0
    # The current throttle of the car
    throttle = float(data["throttle"])
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :].astype(np.float32)

    X = preprocess_input(transformed_image_array)
    base_X = base_model.predict(X)
    base_time = time.time()

    new_steering_angle = float(model.predict(base_X, batch_size=1))
    if new_steering_angle < -1: new_steering_angle = -1
    if new_steering_angle > 1: new_steering_angle = 1

    # Smooth the steering angle. If using smoothing in the data, then alpha can
    # be zero. Even with unsmoothed data, it seems to need to be fairly small
    # anyway, in order to turn sharply enough.
    alpha = 0
    steering_angle = alpha * steering_angle + (1 - alpha) * new_steering_angle

    # Don't go too fast. If we're turning, slow down a bit.
    target_speed = 1 + 12 * (1 - abs(steering_angle))

    # Don't accelerate too much.
    max_throttle = 0.6
    min_throttle = -0.2

    # Choose new throttle based on target speed. I am still not entirely clear
    # on the units for throttle, so this factor is just empirical.
    new_throttle = (target_speed - speed) * 0.1
    if new_throttle < min_throttle: new_throttle = min_throttle
    if new_throttle > max_throttle: new_throttle = max_throttle

    # Update throttle with smoothing. Again empirical.
    beta = 0.9
    throttle = beta * throttle + (1 - beta) * new_throttle

    dt_base = base_time - start_time
    dt = time.time() - start_time
    print('dt: %.3fs\tdt_base: %.3fs\tsa: %.3f\tthrottle=%.3f\tnew throttle=%.3f' % (
        dt, dt_base, steering_angle, throttle, new_throttle))
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    check_for_weights_update()
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
