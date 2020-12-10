# IMPORTS 
import numpy as np

import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
import tensorflow.keras.optimizers as optimizers

import sys, os
sys.path.append(os.path.join('', "coco/"))
from mrcnn import utils
import mrcnn.model as modellib
import coco

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

#COCO MODEL
model_weights = 'mask_rcnn_balloon.h5'

# RCNN
rcnn_model = modellib.MaskRCNN(mode='inference', model_dir='./', config=config)
rcnn_model.load_weights('mask_rcnn_balloon.h5')
rcnn_model.summary()

# CNN


