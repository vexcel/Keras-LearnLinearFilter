# Keras-LearnLinearFilter
**Keras implementation for learning linear filtering from data.**

In this project, a simple single layer convolutional neural network is trained to perform linear filtering using natural image data.
The convolution kernel weights are saved on each epoch during training and visualized. 

The straightforward problem statement allows us to develop the intuition on how the convolution layers operate on the data and also how the convolution layers learn the important features from the data.

As a training data we will utilize the Urban and Natural Scene Categories - dataset by the Computational Visual Cognition Laboratory, Massachusetts Institute of Technology (Oliva, A. & Torralba, A. (2001).Modeling the Shape of the Scene: A Holistic Representation of the Spatial Envelope).

In the image below we can find an example of an original image (left-hand side), a grayscale converted image (middle) and a Sobel filtered image (right-hand side) in x-direction.
![Original, grayscale and Sobel filtered image](OriginalGrayFiltered_sobelx.png?raw=true "Original, grayscale and Sobel filtered image")

The model is trained to map the grayscale converted images to linear filtered versions of the images.
The experiments are carried out for three different filter kernels: the Sobel filter in x-direction, the Sobel filter in y-direction and a 32x32 filter kernel representing a smiley face.

Here, for example, we can see how the network weights converge close to the values of the Sobel filter in x-direction:
![A gif movie of kernel weights converging to Sobel x-direction filter](weightmoviegx.gif?raw=true "A gif movie of kernel weights converging to Sobel x-direction filter")

The forward pass output (left-hand side) and the result of the linear filtering (right-hand side) on a grayscale image are almost identical as can be observed in the image below:

![Output of the neural network and the result of the Sobel filtering in x-direction](PredictedFiltered_sobelx.png?raw=true "Output of the neural network and the result of the Sobel filtering in x-direction")

Linear filtering can be carried out with an arbitrary kernel.
A model with a properly chosen kernel size can learn to perform filtering with a smiley face filter as well:

![A gif movie of kernel weights converging to a smiley face filter](weightmoviesmiley.gif?raw=true "A gif movie of kernel weights converging to a smiley face filter")

###### Disclaimer:
The learning process can also be viewed as an inverse problem in which case we are committing an inverse crime.
To be more careful, the training data should be pre-processed at least by: injecting some noise to the intensity values and rescaling the resolution.
For our purposes of visualizing the temporal development of the convolution weights and building up the intuition behind the operations which the convolution layers perform to the data, the current approach with no effort on trying to avoid the inverse crime should be sufficient.
