#!/usr/bin/env python3
"""Function that builds a projection block as described in Deep Residual Learning for Image Recognition (2015)"""

import tensorflow.keras as K



def projection_block(A_prev, filters, s=2):
    ''' builds a projection block as described in Deep Residual Learning for Image Recognition (2015)'''
    init = K.initializers.he_normal()
    F11, F3, F12 = filters
    
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=1, strides=s, padding='same', kernel_initializer=init)(A_prev)
    batch1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(batch1)
    
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=3, strides=1, padding='same', kernel_initializer=init)(act1)
    batch2 = K.layers.BatchNormalization()(conv2)
    act2 = K.layers.Activation('relu')(batch2)

    conv3 = K.layers.Conv2D(filters=F12, kernel_size=1, strides=1, padding='same', kernel_initializer=init)(act2)
    batch3 = K.layers.BatchNormalization()(conv3)

    conv4 = K.layers.Conv2D(filters=F12, kernel_size=1, strides=s, padding='same', kernel_initializer=init)(A_prev)
    batch4 = K.layers.BatchNormalization()(conv4)

    add = K.layers.Add()([batch3, batch4])
    act3 = K.layers.Activation('relu')(add)

    return act3
