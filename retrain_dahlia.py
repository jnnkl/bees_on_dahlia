#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:52:23 2018

@author: bombus
"""

"""
This file explains how the movie (...) is made. My data put in a folder called data
and which contains subfolders train, validation and test, each of which contains
a folder with or without, where in the folder with are images of parts of the flower 
with an insect and without are images without an insect.  
"""



#loading the modules we use.
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import applications
from keras.layers import Dropout, Flatten, Dense


from keras import backend as K
K.set_image_dim_ordering('th')

from tensorflow import GPUOptions 

#tensorflow can be greedy, so resrict its use.
gpu_opts = GPUOptions(per_process_gpu_memory_fraction=0.35)


# dimensions of our images.
img_width, img_height = 150, 150
PATH='data/'
top_model_weights_path = 'bottleneck_dahlia.h5'
train_data_dir = 'data/train/'
validation_data_dir ='data/validation/'
test_data_dir ='data/test/'

#some hyper parameters
nb_train_samples = 926
nb_validation_samples = 254
epochs = 50
batch_size = 16



#some useful functions
def save_bottleneck_features():
    #this takes the data and processes it trough the VGG16 model with the
    #weight such as in imagenet.
    
    #resccale the images, because that is the convention in tensoflow
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    #get the training data
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    # get the bottle neck features for the training data
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    #save the bottle_neck features
    np.save('bottleneck_train',
        bottleneck_features_train)

    print("done train")
    #get the validation data
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    #get the bottleneck data for the validation data
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    #save the bottleneck feature
    np.save('bottleneck_val',
            bottleneck_features_validation)
    print("done valid")


def train_top_model():
    #this function makes a new model with as input the bottlenecks created by
    #the previous function, trains the model and saves it afterwards.
    train_data = np.load('bottleneck_train.npy')
    train_labels = np.array([0] * (nb_train_samples // 2) +
                            [1] * (nb_train_samples // 2+nb_train_samples%2))

    validation_data = np.load('bottleneck_val.npy')
    validation_labels = np.array([0] * (nb_validation_samples // 2) +
                                 [1] * (nb_validation_samples // 2+nb_validation_samples%2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    print("starting fitting")
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    print("ended fitting and saving")
    
    
save_bottleneck_features()
train_top_model()    