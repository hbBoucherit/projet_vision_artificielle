# IMPORTS 
import numpy as np
# Neural networks dependancies
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
import tensorflow.keras.optimizers as optimizers

# DATA
#MSCOCO


# RCNN

rcnn = MaskRCNN(mode='inference', model_dir='./', config=(NAME="test", GPU_COUNT = 1, IMAGES_PER8GPU = 1, NUM_CLASSES=81))
model = load_weight('mask_rcnn_balloon.h5', by_name=True)
model.summary()

# CNN


