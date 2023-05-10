#!/usr/bin/env python3
""" Module that creates a neural network with keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    input = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            layer = K.layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=L2,
                                   name='dense')(input)
        else:
            layer = K.layers.Dense(layers[i], activation=activations[i],
                                   kernel_regularizer=L2,
                                   name='dense_' + str(i))(layer)
        if i < len(layers) - 1:
            layer = K.layers.Dropout(1 - keep_prob)(layer)
