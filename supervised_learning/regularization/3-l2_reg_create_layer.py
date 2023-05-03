#!/usr/bin/env pytohn3
"""Function that creates a tensorflow layer that includes L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer that includes L2 regularization"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel,
                            kernel_regularizer=l2)
    return layer(prev)
