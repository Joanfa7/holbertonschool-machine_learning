#!/usr/bin/env python3
""" Function that creates a tensorflow layer that includes dropout 
regularization"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that creates a tensorflow layer that includes dropout 
    regularization"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel,
                            kernel_regularizer=dropout)
    return layer(prev)