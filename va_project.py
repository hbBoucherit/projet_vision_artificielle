import sys, time, os, warnings 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from collections import OrderedDict

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

#import keras
#from keras.backend.tensorflow_backend import set_session
#from collections import Counter 

# Location of the Flickr8k images
dataset_image_path ="./flickr8k/Images/"
# Location of the caption file
dataset_text_path  ="./flickr8k/captions.txt" 
# Wanted shape for images
wanted_shape = (224,224,3)

# To obtain the text dataset corresponding to images
df_texts = pd.read_csv(dataset_text_path) #["image","caption"] 
n_img = df_texts.count()/5 # 40455/5 

# To filter image column in df_texts so that images appear only one time
unique_img = pd.unique(df_texts["image"])# 8091 unique images

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

#print(padding(arr, 99, 13).shape)  => (99,13)

# Function to crop images
def crop_center(img):
    cropx, cropy, _ = wanted_shape
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# Images saved to df_images as (224,224,3) for each unique img in the dataset
def load_img_from_ds(image_name):
    #PREPROCESSING
    img =  img_to_array(load_img(dataset_image_path+image_name, target_size=wanted_shape))
    #img = crop_center(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    #print(f"shape img input {image_name} : {img.shape}")
    return img

# Using pretrained model VGG16
base_model = VGG16(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=wanted_shape, pooling=None, classes=1000,
    classifier_activation='softmax'
)
# To freeze VGG16 weights
#feature_extractor.trainable=False

from tensorflow.keras.models import Model
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
model.summary()
fm=[]
for i in range(len(unique_img)):
    img = load_img_from_ds(unique_img[i])
    feature_map = model.predict(img)
    print(feature_map.shape)
    fm.append(feature_map)

fm=np.array(fm)
print(fm.shape)

#print(f"Shape des fm {feature_maps.shape}")
# To obtain the feature maps
feature_maps = OrderedDict()
images = os.listdir(dataset_image_path)
for i, name in enumerate(images):
    filename = dataset_image_path + name
    image = load_img(filename, target_size=wanted_shape)
    #prediction = 
    #feature_maps[name] = 

