# Behaviorial Cloning

![Alt Text](./images/video.gif)

Overview
---

The goal of the project is to use end to end learning to autonomously drive a car in a simulated environment. The input consists of images read from 3 dashboard cameras and the output is the steering angle. The [simulator](https://github.com/udacity/self-driving-car-sim) has two modes - training and autonomous driving. During training, we drive the car around the track and collect data (3 dashboard camera images, steering angle, throttle, brake). We use Keras to train and validate a convolutional neural network to predict the steering angle. In the autonomous mode, only the center camera image is fed into the network and the predicted steering angle is applied. This project was done as part of Udacity's self-driving nanodegree program.


[//]: # (Image References)

[image1]: ./images/model.png "model"
[image2]: ./images/camera_images.png "3 camera images"
[image3]: ./images/flip.png "Flipped Image"
[image4]: ./images/hist1.png "hist before"
[image5]: ./images/hist2.png "hist after"
[image6]: ./images/recovery.gif "Recovery Image"



### Description of files

* `model.py` reads in the data (camera images + steering angles), preprocesses it and trains a model
* `drive.py` drives the car in autonomous mode with the trained model
* `model.h5` contains a trained convolution neural network 

Using [Udacity's simulator](https://github.com/udacity/self-driving-car-sim) and `drive.py`, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model and training

The final model is an amalgamation of the LeNet architecture and a scaled down version of [Nvidia's model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

![alt text][image1]

It's a feedforward network of 3 convolutional layers and 2 fully connected layers. Each convolutional layer is followed by an ELU to introduce nonlinearity and a max pooling layer for subsampling. Dropout layers are used for regularization with a 0.5 probability of retaining neurons. This helps in avoiding overfitting and it also ensembles activations learnt from different neurons. Camera images are resized to 64x64 and are normalized by a Keras lambda layer that mean centers and scales the pixels between -0.5 to 0.5. An Adam optimizer is used with a mean squared error loss function between the predicted and actual steering angle. The dataset was split into 80% training and 20% validation set. The model was trained only with the training set and the validation set was used to tune hyperparameters. The model was trained with 20,000 samples per epoch for a total of 3 epochs. 

### Training data

I used the dataset provided by Udacity. It captures good driving behavior well. That was enough for the car to successfully drive around the lap autonomously. 

*Multiple camera images with angle offset*
During training, images are recorded from the left, center and right dashboard camera. An angle offset of 0.25 degrees is applied to the left camera image and -0.25 degrees to the right camera image. By doing so, we not only feed in more data to the network, but simulate driving near the corners and recovery from it. For a left turn, the left camera image would correspond to a softer left turn and the right camera image would correspond to a harder left turn. 

![alt text][image2]

*Flipping*
Since the lap is anti-clockwise, we have data with mostly left turns. This would cause the model to have a bias towards predicting left turns. During the early stages of tweaking the model, the car would just drive around in a loop in place. One way to solve this would be to collect training data by driving in the opposite direction. I took the lazy route of flipping the images and steering angles to simulate driving clockwise. 

![alt text][image3]

#### Generators

I used Keras generators to yield a batch of data when required. This reduces the memory footprint by loading only a batch of images and steering angles instead of all of them. I used a batch size of 32. On every call to the generator, it would randomly sample from the input data, randomly choose one of the camera images and randomly decide to flip it or not. Random samples are taken from a unifrom distribution. 

#### Image preprocessing

Each image is cropped to get rid of unnecessary pixels in the scene. This allows the network to not only concentrate on relevant features but also speeds up training and testing. For real time driving, we would need the minimize the latency of a forward pass through the network. Images are then resized of 64x64 and changed to the YUV space as suggested in the [Nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

### Data histogram flattening

 Since the track consists of mostly straight sections, the data is biased towards driving straight. It has a high proportion of near 0 degrees steering angles.

![alt text][image4]

This biased the model to only drive straight and not perform well on turns. I divided the steering angles among 25 bins and capped the angles within each bin to 1.8 times the mean sample per bin.  

![alt text][image5]

This helped the model recover well from sharp turns.

![alt text][image6]


### Solution approach

I started with a LeNet architecture and tweaked the model until it started overfitting on a few images. On training with only the center images, the car would not do well on turns and would drive straight off the road. On adding all three camera images, it would take turns, but was biased towards the left turn. At one point, the car would keep going left and end up driving around anti-clockwise in a loop. This problem went away on flipping the images and steering angles to simulate driving towards the right. It would turn better but would not recover if it turned too much. I equalized the histogram by reducing the data with near 0 degrees steering angle. On tweaking the equaliztion a bit, the recovery was better as the model lernt how to handle turns better.

### Thoughts, problems and future work

I've been thinking of simulating end to end learning in a simulated environment like GTA 5, similar to [pygta5](https://github.com/Sentdex/pygta5). This project and Udacity's simulator was well developed and extremely helpful. Kudos to the team for putting in so much hard work and setting up the project so well. 

With the current model, the car drives around the lap multiple times autonomously. But slowly. On maximizing the throttle, even on straight sections, it starts oscillating on turns until it becomes unstable and veers off the track. The model predicts larger steering angles than it should. Maybe reducing the aggressiveness of culling data with near 0 degrees steering angles can help. A lot of times, the car in fact should go straight. Underestimating the prior probability can be detrimental and seems to be so. 
The model doesn't perform well on the second unseen track. Although it's not part of the requirement, I'm interested in the getting it to learn features effectively required for driving. Augmenting the data further by adding in translation, brightness and shadow changes can simulate driving under varied conditions (slopes, sunny, dark, raining, snowing). With more complex scenarios, I'm curious to apply reinforcement learning with CNNs as function approximators to train vehicles in driving on the track as long as possible.
