#!/usr/bin/env python3
''' Converts a laber vector into a one-hot matrix'''

import tensorflow.keras as K


def one_hot(labels, classes=None):
    ''' Converts a laber vector into a one-hot matrix'''
    return K.utils.to_categorical(labels, classes)
