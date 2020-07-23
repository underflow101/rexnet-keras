###########################################################################
# rexnet.py
#
# Dev. Dongwon Paek
# Description: ReXNetV1 Source Code
#              Run this code with $ python rexnet.py on terminal
###########################################################################

import h5py, os, shutil, math

import tensorflow as tf
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD

from hyperparameter import *
from model import ReXNetV1

config = CONFIG()

inputs = tf.keras.Input(shape=(224,224,3), name='modelInput')

model = ReXNetV1(width_mult=0.5)
model.build(input_shape=(None, 224, 224, 3))
model.summary()
model.save_weights('./pretrained/rexnet_0.5x.h5')

print("Saved Model.")



sgd = SGD(lr=LEARNING_RATE, decay=config.weight_decay, momentum=config.momentum, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
print("Done compile")

#model.save('./pretrained')

print("DONE")