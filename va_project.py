import sys, time, os, warnings 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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

# Using pretrained model VGG16
feature_extractor = VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)
# To freeze VGG16 weights
feature_extractor.trainable=False
feature_extractor.summary()

'''
final_model = Sequential([
    feature_extractor
])
final_model.summary()
'''
# To obtain the text dataset corresponding to images
df_texts = pd.read_csv(dataset_text_path) #["image","caption"] 
n_img = df_texts.count()/5 # 40455/5 

# To filter image column in df_texts so that images appear only one time
unique_img = pd.unique(df_texts["image"])# 8091 unique images

# Images saved to df_images as (224,224,3) for each unique img in the dataset
df_images=np.array([np.array(load_img(dataset_image_path+unique_img[i], target_size=wanted_shape)) for i in range(len(n_img))])

# To obtain the feature maps
images = os.listdir(dataset_image_path)
for i, name in enumerate(images):
    filename = dataset_image_path + '/' + name
    image = load_img(filename, target_size=(224,224,3))
    image = img_to_array(image)
    image = preprocess_input(image)
    