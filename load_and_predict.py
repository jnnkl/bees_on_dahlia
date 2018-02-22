#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:10:27 2018

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



img_width, img_height = 150, 150


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")



def load_and_predict(file):
    # loads the file and predict with the model if there is an insect or not
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    #print("x.shape",x.shape)
    prediction=model.predict(x)
    #print("predarrays",prediction)
    pred=0
    if prediction>0.5:
        print("without")
        pred="no"
    else:
        print("with")
        pred="yes"    
    return pred,prediction[0][0]    

