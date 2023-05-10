#!/usr/bin/env python3
""" Module that creates a neural network with keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    input = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            output = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=L2)(input)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=L2)(dropout)
    return K.Model(inputs=input, outputs=output)
