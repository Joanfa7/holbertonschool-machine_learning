#!/usr/bin/env python3
''' Function that trains a model using mini-batch gradient descent '''

import tensorflow.keras as K


def train_model(nx, layers, activations, lambtha, keep_prob):
    ''' Function to train a model '''
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                                     activation=activations[i],
                                     kernel_regularizer=L2,
                                     name='dense'))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=L2,
                                     name='dense_' + str(i)))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
