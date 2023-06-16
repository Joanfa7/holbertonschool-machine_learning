#!/usr/bin/env python3
''' Create a inception_block model '''

import tensorflow.keras as K


def inception_block(A_prev, filters):
    ''' Function that creates an inception block '''
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                            padding='same', activation='relu')(A_prev)
    conv3R = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                             padding='same', activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                            padding='same', activation='relu')(conv3R)
    conv5R = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                             padding='same', activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                            padding='same', activation='relu')(conv5R)
    pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                              padding='same')(A_prev)
    convPP = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                             padding='same', activation='relu')(pool)
    output = K.layers.concatenate((conv1, conv3, conv5, convPP))
    return output
