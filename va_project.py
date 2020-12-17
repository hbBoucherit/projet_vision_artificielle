import sys, time, os, warnings 

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

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
import gensim, logging

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
df_texts = pd.read_csv(dataset_text_path, sep=",") #["image","caption"] 
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


nltk.download('stopwords')
nltk.download('punkt')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english')) 


#Text processing
def process_sentence(sentence):
    #Text preparation
    def remove_stopwords(caption, stop_words=stop_words):
        word_tokens = word_tokenize(caption)  
        filtered_caption = " ".join([w for w in word_tokens if w not in stop_words])
        return filtered_caption

    # Add start and end sequence token
    def add_start_end_seq_token(txt):
        return 'startseq ' + txt + ' endseq'

    def text_pipeline(caption):
        # lowercase
        caption = caption.lower()
        caption = remove_stopwords(caption)
        # remove some punctuations
        caption = caption.replace(".", "")
        caption = caption.replace(",", "")
        # remove numeric values
        caption = re.sub("\d+", "", caption)
        caption = add_start_end_seq_token(caption)
        return caption
    
    return text_pipeline(sentence)


# Change character vector to integer vector
# Embedding
def character_to_integer_vector(df_texts):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df_texts) #list of captions to train on
    #print("size of the dictionary : " + len(tokenizer.word_index) +1)
    return tokenizer.texts_to_sequences(df_texts)


df_texts["cleaned"]=[process_sentence(s) for s in df_texts["caption"]]
df_texts["tokenized"]=character_to_integer_vector(df_texts["cleaned"])
print(df_texts.head(5))


# Word2Vec - si ça ne marche pas, on fait ce qu'a fait Fairyonice
#START WORD2VEC
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# train word2vec on the two sentences
model = gensim.models.Word2Vec([word_tokenize(w) for w in df_texts["cleaned"]], min_count=1, size=4096)

print(model["girl"].shape)
print(model["boy"].shape)
print(f"Similarité : {model.similarity('girl', 'boy')}")
#END WORD2VEC

# ACP pour faire correspondre les dimensions du texte et image
