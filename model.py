# coding:utf-8
"""
DCGAN model 

"""

import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

import numpy as np

# Generator
def build_generator(z_dim):
    
    noise_shape = (z_dim,)
    model = Sequential()

    model.add(Dense(128*16*16, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16,16,128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding='same'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())

    model.add(Conv2D(64, kernel_size=3, padding='same'))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2D(3, kernel_size=3, padding='same'))
    model.add(Activation('tanh'))

    model.summary()
    return model

# Discriminator
def build_discriminator(img_rows, img_cols, channels):
    
    img_shape = (img_rows, img_cols, channels)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(ZeroPadding2D(padding=((0,1), (1,0))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(ZeroPadding2D(padding=((0,1), (1,0))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(1))        
    model.add(Activation('sigmoid'))  

    model.summary()

    return model