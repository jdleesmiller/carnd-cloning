from keras.layers import Input
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3

from common import *

def make_full_model():
    return InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=Input(shape=IMAGE_SHAPE))

def make_cut_model(cut_index):
    """
    Return the first part of the Inception model. Running the whole inception
    model is too slow for real time use on my laptop, and I imagine that the
    later layers may be more specific to imagenet in any case.

    I have arbitrarily chosen to truncate at the "mixed2" layer. The mixing
    layer indexes are:
    28 mixed0
    44 mixed1
    60 mixed2
    70 mixed3
    92 mixed4
    114 mixed5

    There's also layer index 12, which is the first max pooling layer.
    """
    inception = make_full_model()
    return Model(
        input=inception.input,
        output=inception.layers[cut_index].output)
