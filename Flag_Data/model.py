#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[3]:


import os


# In[4]:


#base_dir = os.path.basename(os.getcwd())
#train_dir = os.path.join(base_dir,'train')
#val_dir = os.path.join(base_dir,'val')

train_dir = 'train'
val_dir = 'val'


# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Augmentation
imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.1
)

batch_size = 64

# train dataset 불러오기
trainGen = imageGenerator.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    subset='training',
    class_mode='categorical'
)

# val dataset 불러오기
validationGen = imageGenerator.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    subset='validation',
    class_mode='categorical'
)


# In[6]:


#옮겨 담기
sample_training_images, _ = next(trainGen)

# 이미지를 보여주는 함수
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample_training_images[:5])


# In[7]:


from tensorflow.keras.layers import InputLayer ,Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# model = keras.Sequential([
#     Flatten(input_shape=(64, 64)),
#     Dense(128, activation='relu'),
#     Dense(7, activation='softmax')
# ])

model = keras.Sequential([ 
    InputLayer(input_shape=(64,64,3)),
    Conv2D(16, (3,3), (1,1), padding ='same', activation = 'relu'),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), (1,1), padding ='same', activation = 'relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), (1,1), padding ='same', activation = 'relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[8]:


trainGen.labels[0]


# In[ ]:


history = model.fit(
    trainGen, 
    epochs=5,
    steps_per_epoch=trainGen.samples / 5, 
    validation_data=validationGen,
    validation_steps=trainGen.samples / 5,
)


# In[ ]:


epochs_range = range(epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


test_dir = os.path.join(val_dir,'test')

testGenerator = ImageDataGenerator(
    rescale=1./255
)

testGen = imageGenerator.flow_from_directory(
    test_dir,
    target_size=(64, 64),
)

# loss and accuracy
model.evaluate_generator(testGen)


# In[ ]:


predictions = model.predict(testGen[0])


# In[ ]:


predictions[0]


# In[ ]:


np.argmax(predictions[0])


# In[ ]:


from tensorflow.keras.preprocessing.image import array_to_img

country_index = ['can', 'eng','fra','ger','ita','kor','usa']


# In[ ]:


imgs = testGen.next()
arr = imgs[0][0]
img = array_to_img(arr).resize((128, 128))
plt.imshow(img)
result = model.predict_classes(arr.reshape(1, 64, 64, 3))
print('예측: {}'.format(country_index[result[0]]))
print('정답: {}'.format(country_index[np.argmax(imgs[1][0])]))


# In[ ]:


# Visualization - not complete
def plot_image(i, prediction, true_label, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if prediction == true_label:
    color = 'blue'
    else:
    color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# In[ ]:


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    imgs = testGen.next()
    arr = imgs[0][0]
    img = array_to_img(arr).resize((128, 128))
    prediction = model.predict_classes(arr.reshape(1, 64, 64, 3))
    
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, country_index[result[0]],country_index[np.argmax(imgs[1][0])] , img)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_arrayplot_image(i, country_index[result[0]],country_index[np.argmax(imgs[1][0])] , img)
plt.show()

