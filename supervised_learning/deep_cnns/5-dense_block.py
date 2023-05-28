#!/usr/bin/env python3
""" dense block model """

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block as described in
    Densely Connected  Convolutional Networks"""
    init = K.initializers.he_normal(seed=None)
    for i in range(layers):
        batch1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation("relu")(batch1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding="same",
            kernel_initializer=init,
        )(act1)
        batch2 = K.layers.BatchNormalization(axis=3)(conv1)
        act2 = K.layers.Activation("relu")(batch2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding="same",
            kernel_initializer=init)(act2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate
    return X, nb_filters
