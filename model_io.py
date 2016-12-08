from keras.models import model_from_json

from common import *

def save_model(json_file, weights_file, model):
    with open(json_file, 'w') as model_file:
        model_file.write(model.to_json())
    model.save_weights(weights_file)

def load_model(json_file, weights_file):
    with open(json_file, 'r') as jfile:
        model = model_from_json(jfile.read())
    model.load_weights(weights_file)
    return model

def load_base_model():
    return load_model(BASE_MODEL_JSON_FILE, BASE_MODEL_WEIGHTS_FILE)

