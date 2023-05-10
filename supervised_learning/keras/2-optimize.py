#!/usr/bin/env python3
''' sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics'''

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''
        @network: model to optimize
        @alpha: learning rate
        @beta1: first Adam optimization parameter
        @beta2: second Adam optimization parameter
        Returns: None
    '''
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
