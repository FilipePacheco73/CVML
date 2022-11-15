# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 07:30:39 2021

@author: Z52XXR7

Code to give an example on how to run a CVML project with TensorFlow and Keras

"""

# Libraries - Imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Load dataset from Keras
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# Take the type of data
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# Check the shape of the data
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# Take a look into the first image
img = plt.imshow(x_train[0])

# Exhibit the label of the image
print('The type of imagem is:', y_train[0])

# Create the classification of data
classification = ['airplane','automobile','bird','cat','deer','dog','frog',
                  'horse','ship','truck']
print('The image class is:', classification[y_train[0][0]])

# Convert the labels into One-Hot encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Normalize the intensity of the pixels
x_train = x_train/255
x_test = x_test/255

# Create the Architecture of the ANN - Artificial Neural Network
model = Sequential()
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3))) # First Convolutional Layer
model.add(MaxPooling2D(pool_size = (2,2))) # First Pooling Layer - Filter
model.add(Conv2D(32,(5,5),activation='relu')) # Second Convolutional Layer
model.add(MaxPooling2D(pool_size = (2,2))) # Second Pooling Layer - Filter
model.add(Flatten()) # Linearize the data to give as an input to Neural Network
model.add(Dense(1000,activation='relu')) # ANN First Layer
model.add(Dropout(0.5)) # Reduce the output size in 50%
model.add(Dense(500,activation='relu')) # ANN Second Layer
model.add(Dropout(0.5)) # Reduce the output size in 50%
model.add(Dense(250,activation='relu')) # ANN Third Layer
model.add(Dense(10,activation='softmax')) # Output Layer with Softmax activation function

# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# Train the model
hist = model.fit(x_train,y_train_one_hot,
                 batch_size = 256,
                 epochs = 50,
                 validation_split = 0.2)

# Evaluate the model with one data test
model.evaluate(x_test,y_test_one_hot)[1]

# Verify the assertiveness of the model
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper left')

# Visualize the Objective Function - Loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper right')

# Test the model with a new picture - never seen before
new_image = plt.imread('cat.jpg') # Let a imagem into the same folder of this python file
new_image = resize(new_image,(32,32,3))
img = plt.imshow(new_image)

predictions = model.predict(np.array([new_image]))
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

print('\n')
for i in range(10):
    print(classification[list_index[i]], ':', round(predictions[0][list_index[i]]*100,2), '%')