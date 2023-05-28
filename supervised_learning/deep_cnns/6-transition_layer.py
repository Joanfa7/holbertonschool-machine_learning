#!/usr/bin/env python3
''' transition layer '''

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    ''' Function that builds a transition layer as described in
        Densely Connected  Convolutional Networks'''
    init = K.initializers.he_normal(seed=None)
    nfilter = int(nb_filters * compression)
    batch1 = K.layers.BatchNormalization(axis=3)(X)
    act1 = K.layers.Activation("relu")(batch1)
    conv1 = K.layers.Conv2D(
        filters=nfilter,
        kernel_size=1,
        padding="same",
        kernel_initializer=init,
    )(act1)
    avg1 = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding="same",
    )(conv1)
    return avg1, nfilter
