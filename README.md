# Self-Driving Car Nanodegree: Behavioral Cloning

This repo contains the code for this [talk and blog post](https://jdlm.info/articles/2019/01/01/driverless-race-car-deep-learning.html).

This was originally an assignment submitted as part of my Udacity Self-Driving Car Engineer Nanodegree, in 2016.

## Results

Please see [README.ipynb](README.ipynb) for the main code and plots.

The notebook calls several python files, some of which may be of interest:

- [the preprocessing code](preprocess.py)
- [the model definition](model.py)
- [the controller than drives the car](drive.py)

## Installation

If you want to run this yourself:

1. This code relies on keras 1.x --- it will not work with newer versions of Keras.

1. You will need to download the [Udacity simulator](https://github.com/udacity/self-driving-car-sim/releases) if you want to run the code.

1. I have not included training data in the repo, so you will also need to collect your own training data, as described in the blog post. This goes under a directory called `data`.

1. This was written against anaconda; most of its dependencies ship by default with anaconda, but there are some additional dependencies you will need:

   ```
   conda install -c conda-forge flask-socketio
   conda install -c conda-forge eventlet
   ```
