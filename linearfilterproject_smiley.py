# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:33:10 2018

@author: admin
"""

import numpy as np
import os
import cv2

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras import optimizers

import matplotlib.pyplot as plt

def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def filter_image(img, kernel):
    # Perform filtering to the input image
    convolved = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
    return convolved

def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)
    img = img/255.0
    return img

def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img


def load_kernel(path='kernels/smiley_32.png'):
    # Load a kernel image into numpy array
    kernel = cv2.imread(path, 0)
    kernel = cv2.bitwise_not(kernel).astype(np.float32)
    # Normalize according to the L2-norm
    kernel = kernel/np.linalg.norm(kernel)
    return kernel

    
# Get the paths to the training images
data_dir = 'data'

folderpaths = [os.path.join(data_dir, o) for o in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,o))]
imagepaths = []

for folderpath in folderpaths:
    temppaths = [os.path.join(folderpath, fname) for fname in os.listdir(folderpath) if fname.endswith('.jpg')]
    imagepaths += temppaths
    
# Load and pre-process the training data
images = []
grayimages = []
filteredimages = []

kernel = load_kernel()

np.random.shuffle(imagepaths)
for imagepath in imagepaths:
    print(imagepath)
    img = cv2.imread(imagepath).astype(np.float32)
    img = normalize_image255(img)
    gray_img = make_grayscale(img)
    filtered_img = filter_image(gray_img, kernel)
    
    images.append(img)
    grayimages.append(gray_img)
    filteredimages.append(filtered_img)
    
images = np.array(images, dtype='float32')
grayimages = np.array(grayimages, dtype='float32')
filteredimages = np.array(filteredimages, dtype='float32')

# Expand the image dimension to conform with the shape required by keras and tensorflow, inputshape=(..., h, w, nchannels).
grayimages = np.expand_dims(grayimages, -1)
filteredimages = np.expand_dims(filteredimages, -1)

print("images shape: {}".format(images.shape))
print("grayimages shape: {}".format(grayimages.shape))
print("filteredimages shape: {}".format(filteredimages.shape))

print("Kernel norm: {}".format(np.linalg.norm(kernel)))

# Visualize an arbitrary image and the filtered version of it
margin_img = np.ones(shape=(256, 10, 3))
combined_image = np.hstack((img, margin_img, np.dstack((gray_img,)*3), margin_img, np.dstack((normalize_image(filtered_img),)*3)))

cv2.imwrite('OriginalGrayFiltered_smiley.png', (255.0*combined_image).astype(np.uint8))

#%%

plt.figure()
plt.imshow(kernel, cmap='gray')
plt.colorbar()
plt.title('Original convolution kernel')
plt.axis('off')
plt.show()
plt.savefig('smileykernel.png')

#%%

input_height, input_width = gray_img.shape

def linearcnn_model():
    # Returns a convolutional neural network model with a single linear convolution layer
    model = Sequential()
    model.add(Conv2D(1, (32,32), padding='same', input_shape=(input_height, input_width, 1)))
    return model


model = linearcnn_model()
adam = optimizers.Adam(lr=1e-3)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

number_of_epochs = 100
loss = []
val_loss = []
convweights = []


for epoch in range(number_of_epochs):
    history_temp = model.fit(grayimages, filteredimages,
                        batch_size=1,
                        epochs=1,
                        validation_split=0.2)
    loss.append(history_temp.history['loss'][0])
    val_loss.append(history_temp.history['val_loss'][0])
    convweights.append(model.layers[0].get_weights()[0].squeeze())

#%%
# Plot the training and validation losses
plt.close('all')
    
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.show()
plt.savefig('trainingvalidationlosssmiley.png')



plt.figure()
plt.imshow(convweights[-1], cmap='gray')
plt.colorbar()
plt.show()

#%%

def visualize_matrix(M, epoch=1):
    """
    Create a visualization of an arbitrary matrix.
    """
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    
    title = "Epoch {}".format(epoch)
    ax.set_title(title, fontsize=20)
    
    matshowplt = ax.matshow(M, cmap='gray', vmin=-0.1, vmax=0.15)
    cbar = plt.colorbar(matshowplt, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18) 
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('Weight value', rotation=270, fontsize=20)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    
    return fig

#%%
savefolder = 'images/'

for i in range(len(convweights)):
    savepath = savefolder+'weightfigure'+str(i)+'.png'
    print(savepath)
    M = convweights[i]
    fig = visualize_matrix(M, epoch=i+1)
    fig.savefig(savepath)
    plt.close(fig)
    
#%%
    
import imageio
from natsort import natsorted

# Get the paths to the convolution layer weight images
image_dir = 'images/'

imagepaths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('png')]
imagepaths = natsorted(imagepaths)

with imageio.get_writer('weightmoviesmiley.gif', mode='I') as writer:
    for impath in imagepaths:
        image = imageio.imread(impath)
        writer.append_data(image)
        
#%%
predicted_img = model.predict(np.array([np.expand_dims(gray_img, -1)]))
predicted_img = predicted_img.squeeze()
        
margin_img = np.ones(shape=(256, 10, 3))
combined_image = np.hstack((np.dstack((normalize_image(predicted_img),)*3), margin_img, np.dstack((normalize_image(filtered_img),)*3)))

cv2.imwrite('PredictedFiltered_smiley.png', (255.0*combined_image).astype(np.uint8))
        
