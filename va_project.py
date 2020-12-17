import sys, time, os, warnings 

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from collections import OrderedDict

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

import re

#import keras
#from keras.backend.tensorflow_backend import set_session
#from collections import Counter 

# Location of the Flickr8k images
dataset_image_path ="flickr8k/Images/"
# Location of the caption file
dataset_text_path  ="flickr8k/captions.txt" 
# Wanted shape for images
wanted_shape = (224,224,3)

# To obtain the text dataset corresponding to images
df_texts = pd.read_csv(dataset_text_path) #["image","caption"] 
n_img = df_texts.count()/5 # 40455/5 

# To filter image column in df_texts so that images appear only one time
unique_img = pd.unique(df_texts["image"])# 8091 unique images

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
'''
base_model = VGG16(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=wanted_shape, pooling=None, classes=1000,
    classifier_activation='softmax'
)

# Feature extraction
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
model.summary()
'''

charge_image, one_by_one = False, False
# To obtain the feature maps
if charge_image:
    feature_maps = np.array([model.predict(load_img_from_ds(unique_img[i])) for i in range(len(unique_img))])
    print(f"Shape des fm {feature_maps.shape}")
elif one_by_one:
    feature_maps=[]
    #for i in range(len(unique_img)):
    #    img = load_img_from_ds(unique_img[i])
    #    feature_map = model.predict(img)
    #    print(feature_map.shape)
    #    feature_maps.append(feature_map)
    #feature_maps=np.array(feature_maps)

sentence = ["I have 3 dogs, BUT I prefer cats.", "Little Julien."]

nltk.download('stopwords')
nltk.download('punkt')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english')) 


#Text preparation
def text_pipeline(df_caption):
    for caption in df_caption:
        # lowercase
        caption = caption.lower()
        # remove some punctuations
        caption = caption.replace(".", "")
        caption = caption.replace(",", "")
        # remove numeric values
        caption = re.sub("\d+", "", caption)

    return df_caption

sentence=text_pipeline(sentence)

def remove_stopwords(df_caption, stop_words=stop_words):
    word_tokens = word_tokenize(caption)  
    filtered_caption = "".join([w for w in word_tokens if w not in stop_words])
    return filtered_caption

sentence = remove_stopwords(sentence)
print(f"Stop words elimination : {sentence}")

# Add start and end sequence token
def add_start_end_seq_token(df_captions):
    captions = []
    for txt in df_captions:
        txt = 'startseq ' + txt + ' endseq'
        captions.append(txt)
    return captions

print(add_start_end_seq_token(sentence))
df_texts0 = df_texts.copy()
df_texts0["caption"] = add_start_end_seq_token(df_texts["caption"])
df_texts0.head(5)

# Change character vector to integer vector
tokenizer = Tokenizer(nb_words=10000)
tokenizer.fit_on_texts(df_texts0["caption"])



#remove stop words
'''
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
Stop words elimination : I 3 dogs , BUT I prefer cats .
['startseq i endseq', 'startseq   endseq', 'startseq   endseq', 'startseq d endseq', 'startseq o endseq', 'startseq g endseq', 'startseq s endseq', 'startseq   endseq', 'startseq   endseq', 'startseq b endseq', 'startseq u endseq', 'startseq t endseq', 'startseq   endseq', 'startseq i endseq', 'startseq   endseq', 'startseq p endseq', 'startseq r endseq', 'startseq e endseq', 'startseq f endseq', 'startseq e endseq', 'startseq r endseq', 'startseq   endseq', 'startseq c endseq', 'startseq a endseq', 'startseq t endseq', 'startseq s endseq', 'startseq   endseq']
'''