# coding: utf-8

"""
Before starting:
Download the training data,
save them in the same directory where
this file is located at.
"""

# importing modules
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from PIL import ImageFile


from __future__ import print_function

import keras
from keras import backend as K
from IPython.display import SVG
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Model, Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.utils.vis_utils import model_to_dot
from keras.utils.generic_utils import get_custom_objects
from keras.preprocessing.image import ImageDataGenerator

# importing layers that consist our neural net
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D
from keras.layers import Activation, Add, BatchNormalization, Dropout

# Set the CNN model
model = Sequential()

model.add(Conv2D(activation='relu', input_shape=(64, 64, 3), filters=64, kernel_size=(3, 3), padding="SAME",
                     strides=(1, 1)))
model.add(Conv2D(activation='relu', filters=64, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(activation='relu', filters=512, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(Conv2D(activation='relu', filters=512, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(activation='relu', filters=256, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(Conv2D(activation='relu', filters=256, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(Conv2D(activation='relu', filters=256, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(activation='relu', filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(Conv2D(activation='relu', filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax', kernel_initializer='uniform'))

model.summary()

# Define the optimizer
optimizer = Adam(lr=0.001)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

# Set a learning rate annealer
# If accuracy is not improved after 3 epochs, then reduce the learning rate to its half
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Data Augmentation
train_datagen = ImageDataGenerator(
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')


test_datagen = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

# path = '10_country_train_1(2)/train/'
train_generator = train_datagen.flow_from_directory(
                                                    '10_country_train_2/train/',
                                                    target_size=(64, 64),
                                                    batch_size=30,
                                                    color_mode='rgb',
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                                                    '10_country_test/test',
                                                    target_size=(64, 64),
                                                    batch_size=30,
                                                    color_mode='rgb',
                                                    class_mode='categorical')

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Train model for 50 epoch
history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=4283 /30,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=50,
                    verbose=1,
                    callbacks=[learning_rate_reduction])

# Training and validation curves
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

test = image.load_img('10_country_test/test/Argentina/0 (1).jpg',
                      target_size=(64, 64))

# Check New Image
test = image.img_to_array(test)
test = np.expand_dims(test, axis=0)

print(model.predict(test))
print(train_generator.class_indices)