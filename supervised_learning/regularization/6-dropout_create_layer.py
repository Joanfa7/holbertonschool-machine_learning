#!/usr/bin/env python3
""" 6. Create a Layer with Dropout """

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """function that creates a layer of a neural network using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)
