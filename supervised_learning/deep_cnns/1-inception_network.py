#!/usr/bin/env python3
"""
Module: Inception Network
"""
from tensorflow.keras import Input, initializers, layers, Model
from '0-inception_block' import inception_block


def build_inception_network():
    """
    Builds the inception network as described in 'Going Deeper with Convolutions' (2014)
    Returns:
        Keras model
    """
    input_shape = (224, 224, 3)
    init = initializers.he_normal()
    activation = 'relu'
    filters = [64, 192, [64, 96, 128, 16, 32, 32], [128, 128, 192, 32, 96, 64], [192, 96, 208, 16, 48, 64],
               [160, 112, 224, 24, 64, 64], [128, 128, 256, 24, 64, 64], [112, 144, 288, 32, 64, 64],
               [256, 160, 320, 32, 128, 128], [256, 160, 320, 32, 128, 128], [384, 192, 384, 48, 128, 128]]
    
    model_input = Input(shape=input_shape)
    x = model_input
    
    for i, filter in enumerate(filters):
        if isinstance(filter, list):
            x = inception_block(x, filter)
            if i in {3, 6, 9}:
                x = layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2), padding='same')(x)
        else:
            x = layers.Conv2D(filters=filter, kernel_size=7 if i == 0 else 3, strides=(2, 2) if i == 0 else (1, 1),
                              padding='same', activation=activation, kernel_initializer=init)(x)
            if i in {0, 1}:
                x = layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2), padding='same')(x)
    
    x = layers.AveragePooling2D(pool_size=[7, 7], strides=(1, 1), padding='valid')(x)
    x = layers.Dropout(.4)(x)
    x = layers.Dense(1000, activation='softmax', kernel_initializer=init)(x)
    
    model = Model(inputs=model_input, outputs=x)
    
    return model

