#!/usr/bin/env python3
''' Forward Propagation'''

import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""
    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy