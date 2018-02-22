#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:59:18 2018

@author: bombus
"""




import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model,load_model,model_from_json
from keras.layers import Dropout, Flatten, Dense,Input
from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from keras import backend as K
K.set_image_dim_ordering('th')
import os
from tensorflow import GPUOptions 

#tensorflow can be greedy, so resrict its use.
gpu_opts = GPUOptions(per_process_gpu_memory_fraction=0.35)
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output




# path to the model weights files.

top_model_weights_path = 'bottleneck_dahlia.h5'

img_width, img_height = 150, 150
PATH='data/'
top_model_weights_path = 'bottleneck_dahlia.h5'
train_data_dir = 'data/train/'
validation_data_dir ='data/validation/'
test_data_dir ='data/test/'


nb_train_samples = 926
nb_validation_samples = 254

epochs = 20
batch_size = 5

input_tensor = Input(shape=(3,img_width,img_height))
base_model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights(top_model_weights_path)
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# set the first 15 layers 
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

#train the model
hist = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=30,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

#the accuracies 
accuracies=hist.history['acc']
val_accuracies=hist.history['val_acc']

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# get to see which index is related to each class
print(validation_generator.class_indices)
