#!/usr/bin/env python3
""" DenseNet-121 """

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def transition_layer(X, nb_filters, compression):
    '''Function that builds a transition layer as described in Densely'''
    init = K.initializers.he_normal()
    batch1 = K.layers.BatchNormalization(axis=3)(X)
    act1 = K.layers.Activation('relu')(batch1)
    conv1 = K.layers.Conv2D(filters=int(nb_filters * compression),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=init)(act1)
    avg1 = K.layers.AveragePooling2D(pool_size=2,
                                        strides=2,
                                        padding='same')(conv1)
    dense1, nb_filters = dense_block(avg1, nb_filters, growth_rate, 16)
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)
    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 16)
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)
    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 16)
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)
    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)
    avg2 = K.layers.AveragePooling2D(pool_size=7,
                                        strides=None,
                                        padding='same')(dense4)
    output = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=init)(avg2)
    model = K.models.Model(inputs=X, outputs=output)
    return model



  