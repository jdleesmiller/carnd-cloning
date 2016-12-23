# Notes

## Installation

```
conda install -c conda-forge flask-socketio
conda install -c conda-forge eventlet
```

## Basics

Image size is 320x160 for each camera.

I expect that my steering and throttle inputs will need some smoothing. Probably just a simple exponential smoother would do it.

Maybe I would also need to do some smoothing on the outputs in `drive.py`?

If I use transfer learning, what preprocessing would be required? Are they included in the keras model somehow?
A: Yes: https://keras.io/applications/

Convolution certainly seems appropriate. We want some translational invariance in order to recognise a road even if it is not in the middle of the image. Recognising a road seems to me to suggest looking for 'smoothness' (that said, the simulator actually has a pretty rough-looking road); I think some large-ish convolutions will be needed to detect smooth areas.

Simplest thing first: define a small neural net, get it training, and see if we can make it around.

I feel like even an ad-hoc approach using the side cameras would be worthwhile --- basically just a constant steering bias of say 2 degrees. It triples the amount of training data. But, let's start with just the central camera.

Another idea: the speed and desired steering angle are not independent: you should not steer sharply when speed is high. We do know the speed, but the network does not know it as an input. Can we feed in the speed as a parallel input? One for future investigation. Another way of looking at it would be to train the network to also output the safe speed based on the speed training data and then calculate the throttle from that.

## The NVIDIA Paper

They suggest using 1/r for r = radius of the turn as the steering angle signal.
The values from the sim range from -1 to 1 for -25 deg to 25deg.

"The steering label for transformed images is adjusted to one that would steer the vehicle back
to the desired location and orientation in two seconds."

They used mean squared error on the steering angle to train.

They used YUV color space and a normalization layer. Then three 5x5 convolutions (2x2 stride), then 3x3 convolutions (1x1 stride), then flattening (1164 inputs) then 3 fully connected layers (100, 50 and 10 neurons). 

"To remove a bias towards driving straight the training data includes a higher proportion of frames
that represent road curves."

They plotted the activations of the first two feature map layers, an they clearly showed detection of road edges.

## The Inception Paper

https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf

"The main idea of the Inception architecture is to consider
how an optimal local sparse structure of a convolutional vision
network can be approximated and covered by readily
available dense components."

Theoretical and biological considerations suggest that sparsity of connections between layers is better, but sparse matrix math is much more computationally expensive than GPU-accelerated dense matrix math.

They use 1x1 convolutions to reduce the number of features (channels) in the image before applying expensive 3x3 and 5x5 convolutions.

## First Submission Model Selection Notes

```
# The ones with lowest MAE show a lot of weave.

# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'nb_filter': 16, 'l2_weight': 0.005, 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0486434253 30
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 4, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0497479700963 14
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 8, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0505109161774 15
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 64, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0506108435425 25

# PASS... but failed when double checked
# {'version': 5, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 32, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0633698049717 6

# PASS: a bit of weave early on, but recovered
# {'nb_filter': 16, 'nb_epoch': 30, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0603575362906 8

# PASS: a bit tight on the turnout turn and some weave, but pretty good.
# {'nb_filter': 8, 'nb_epoch': 30, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.060572350685 11

# A bit better but still weaves a lot
# 'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 8, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0507841391891 30

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 8, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0520709987096 14

# Made it to the turnout; pretty stable
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 4, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0539738734781 11

# Weaves, but made it almost to the turnout.
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 32, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0546632729836 11

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 64, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0547469570021 15

# Weaves
# {'version': 5, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 4, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0563107190402 11

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 32, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0564398219643 13

# Weaves
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 64, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0569417894736 19

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'nb_filter': 16, 'l2_weight': 0.005, 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0572153125485 12

# Weaves
# {'version': 5, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 8, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0572628182225 10

# Made it to the turnout; pretty stable.
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 8, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.057351532648 11

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 4, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0574356182614 14

# Weaves
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'nb_filter': 16, 'l2_weight': 0.01, 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0575831151575 10

# Weaves
# {'version': 5, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'nb_filter': 16, 'l2_weight': 0.005, 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0576822490848 7

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'nb_filter': 16, 'l2_weight': 0.01, 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0581040763912 10

# End of bridge, but weaves
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 4, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0583697665036 8

# Weaves
# {'version': 5, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 32, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0585662010462 7

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 64, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0586640624298 19

# Weaves
# {'version': 5, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 4, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0604852065088 11

# Weaves
# {'version': 5, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.01, 'nb_filter': 32, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0624602926608 6

# Some weave, did not turn at turnout.
# {'nb_filter': 4, 'nb_epoch': 30, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0530055659274 25

# Some weave, did not turn at turnout.
# {'nb_filter': 16, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.01} 0.0536798351864 30

# Pretty good but hit edge near turnout. Marginal pass.
# {'nb_filter': 8, 'nb_epoch': 30, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0564248780002 13

# Pretty good, did not turn at turnout.
# {'nb_filter': 4, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0581461225773 22

# Lost it on the bridge
# {'nb_filter': 4, 'nb_epoch': 30, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0594900304391 14

# Pretty good, did not turn at turnout.
# {'nb_filter': 16, 'nb_epoch': 30, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0595571538155 11

# Pretty good, did not turn at turnout.
# {'nb_filter': 8, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.01} 0.0601387542259 14

# Pretty good, did not turn at turnout.
# {'nb_filter': 64, 'nb_epoch': 30, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0617620240928 15

# Pretty good, did not turn at turnout.
# {'nb_filter': 32, 'nb_epoch': 30, 'nb_hidden': 32, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0625045372937 8

# A bit of weave, did not turn at turnout.
# {'nb_filter': 16, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0627611787279 10

# A bit of weave, did not turn at turnout.
# {'nb_filter': 32, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0627763003509 16

# A bit of weave, did not turn at turnout.
# {'nb_filter': 8, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0632025661513 10

# A bit of weave, did not turn at turnout.
# {'nb_filter': 64, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.01} 0.0632899765945 9

# Pretty good, did not turn at turnout.
# {'nb_filter': 64, 'nb_epoch': 30, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0649442658476 11

# Weaves.
# {'nb_filter': 64, 'nb_epoch': 30, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.0674919454212 9

# Weaves
# {'nb_filter': 32, 'nb_epoch': 30, 'nb_hidden': 64, 'side_camera_bias': 0.06, 'version': 5, 'optimizer': 'adam', 'batch_size': 128, 'label_column': 'smooth_steering_angle_gaussian_3', 'l2_weight': 0.02} 0.067675512434 5

# Weaves
# {'version': 5, 'nb_hidden': 128, 'side_camera_bias': 0.06, 'batch_size': 128, 'l2_weight': 0.005, 'nb_filter': 64, 'label_column': 'smooth_steering_angle_gaussian_3', 'nb_epoch': 30, 'optimizer': 'adam'} 0.0645840176195 9
```

