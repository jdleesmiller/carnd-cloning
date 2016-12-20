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

## Resubmission

- Tried adding the udacity data into the same model that worked last time.
  Subjectively, performance does seem a little better.
- Suspect that the failure in the first submission was due to a slow laptop ---
  failures look similar to what I get on my older laptop.
- So added timing info into drive.py:

   ```
   dt: 0.177s	dt_base: 0.174s	sa: 0.014	throttle=0.172	new throttle=0.315
   dt: 0.174s	dt_base: 0.169s	sa: 0.014	throttle=0.179	new throttle=0.311
   dt: 0.167s	dt_base: 0.165s	sa: 0.014	throttle=0.179	new throttle=0.311
   dt: 0.151s	dt_base: 0.147s	sa: 0.038	throttle=0.173	new throttle=0.221
   dt: 0.167s	dt_base: 0.165s	sa: 0.033	throttle=0.177	new throttle=0.225
   dt: 0.163s	dt_base: 0.160s	sa: 0.033	throttle=0.177	new throttle=0.225
   dt: 0.152s	dt_base: 0.150s	sa: 0.039	throttle=0.177	new throttle=0.155
   dt: 0.163s	dt_base: 0.161s	sa: 0.040	throttle=0.171	new throttle=0.152
   dt: 0.165s	dt_base: 0.162s	sa: 0.040	throttle=0.171	new throttle=0.152
   dt: 0.156s	dt_base: 0.153s	sa: 0.025	throttle=0.169	new throttle=0.095
   dt: 0.162s	dt_base: 0.159s	sa: 0.018	throttle=0.174	new throttle=0.146
   ```

   That shows that essentially the whole time is the base model prediction.
- So, trying it out with the first 28 inception layers instead of 44, to see
  whether that still gives OK performance with reduced calculation.
