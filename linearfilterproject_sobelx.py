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

def filter_image_sobelx(img):
    # Perform filtering to the input image
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    return sobelx

def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)
    img = img/255.0
    return img

def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img

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

np.random.shuffle(imagepaths)
for imagepath in imagepaths:
    print(imagepath)
    img = cv2.imread(imagepath).astype(np.float32)
    img = normalize_image255(img)
    gray_img = make_grayscale(img)
    filtered_img = filter_image_sobelx(gray_img)
    
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

# Visualize an arbitrary image and the filtered version of it
margin_img = np.ones(shape=(256, 10, 3))
combined_image = np.hstack((img, margin_img, np.dstack((gray_img,)*3), margin_img, np.dstack((normalize_image(filtered_img),)*3)))

cv2.imwrite('OriginalGrayFiltered_sobelx.png', (255.0*combined_image).astype(np.uint8))

#%%

input_height, input_width = gray_img.shape

def linearcnn_model():
    # Returns a convolutional neural network model with a single linear convolution layer
    model = Sequential()
    model.add(Conv2D(1, (3,3), padding='same', input_shape=(input_height, input_width, 1)))
    return model


model = linearcnn_model()
sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

number_of_epochs = 100
loss = []
val_loss = []
convweights = []


for epoch in range(number_of_epochs):
    history_temp = model.fit(grayimages, filteredimages,
                        batch_size=4,
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
plt.savefig('trainingvalidationlossgx.png')

# Visualize the convolution weight at the last epoch
plt.figure()
plt.imshow(convweights[-1], cmap='gray')
plt.colorbar()
plt.show()

#%%

def visualize_matrix(M, epoch=1):
    """
    Create a visualization of an arbitrary matrix.
    """
    fig = plt.figure(figsize=(10,5))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    
    title = "Epoch {}".format(epoch)
    fig.suptitle(title, fontsize=20)
    
    height, width = M.shape
    Mud = np.flipud(M) # Now the i-index complies with matplotlib axes
    coordinates = [(i,j) for i in range(height) for j in range(width)]
    print(coordinates)
    for coordinate in coordinates:
        i,j = coordinate
        value = np.round(Mud[i,j], decimals=2)
        relcoordinate = (j/float(width), i/float(height))
        ax1.annotate(value, relcoordinate, ha='left', va='center',
                     size=22, alpha=0.7, family='serif')
        
    padding = 0.25
    wmargin = (width-1)/float(width) + padding
    hmargin = (height-1)/float(height) + padding
    
    hcenter = np.median(range(height))/float(height)
    print(hcenter)
    hcenter = hcenter + 0.015 # Offset due to the character alignment
    
    bracket_d = 0.4
    bracket_b = 0.05
    bracket_paddingl = 0.05
    bracket_paddingr = -0.05
    
    ax1.plot([-bracket_paddingl, -bracket_paddingl],[hcenter-bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([-bracket_paddingl, -bracket_paddingl+bracket_b], [hcenter-bracket_d, hcenter-bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([-bracket_paddingl, -bracket_paddingl+bracket_b], [hcenter+bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    
    ax1.plot([wmargin-bracket_paddingr, wmargin-bracket_paddingr],[hcenter-bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([wmargin-bracket_paddingr-bracket_b, wmargin-bracket_paddingr], [hcenter-bracket_d, hcenter-bracket_d], 'k-', lw=2, alpha=0.7)
    ax1.plot([wmargin-bracket_paddingr-bracket_b, wmargin-bracket_paddingr], [hcenter+bracket_d, hcenter+bracket_d], 'k-', lw=2, alpha=0.7)
    
    ax1.set_xlim([-padding, wmargin+0.06])
    ax1.set_ylim([-padding, hmargin])
    
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.axis('off')
    
    matshowplt = ax2.matshow(M, cmap='gray', vmin=-2, vmax=2)
    cbar = plt.colorbar(matshowplt, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18) 
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('Weight value', rotation=270, fontsize=20)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    
    return fig

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

# Get the paths to the convolution weight visualization images
image_dir = 'images/'

imagepaths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
imagepaths = natsorted(imagepaths)

with imageio.get_writer('weightmoviegx.gif', mode='I') as writer:
    for impath in imagepaths:
        image = imageio.imread(impath)
        writer.append_data(image)

        
#%%
predicted_img = model.predict(np.array([np.expand_dims(gray_img, -1)]))
predicted_img = predicted_img.squeeze()
        
margin_img = np.ones(shape=(256, 10, 3))
combined_image = np.hstack((np.dstack((normalize_image(predicted_img),)*3), margin_img, np.dstack((normalize_image(filtered_img),)*3)))

cv2.imwrite('PredictedFiltered_sobelx.png', (255.0*combined_image).astype(np.uint8))
        
