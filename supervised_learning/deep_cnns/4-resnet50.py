#!/usr/bin/env python3
''' ResNet-50 model '''

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    ''' Function that builds the ResNet-50 architecture as described
        in Deep Residual Learning for Image Recognition (2015) '''
    init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        kernerl_initializer=init)(X)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(batch1)
    pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(act1)
    proj1 = projection_block(pool1, [64, 64, 256], 1)
    iden1 = identity_block(proj1, [64, 64, 256])
    iden2 = identity_block(iden1, [64, 64, 256])
    proj2 = projection_block(iden2, [128, 128, 512])
    iden3 = identity_block(proj2, [128, 128, 512])
    iden4 = identity_block(iden3, [128, 128, 512])
    iden5 = identity_block(iden4, [128, 128, 512])
    proj3 = projection_block(iden5, [256, 256, 1024])
    iden6 = identity_block(proj3, [256, 256, 1024])
    iden7 = identity_block(iden6, [256, 256, 1024])
    iden8 = identity_block(iden7, [256, 256, 1024])
    iden9 = identity_block(iden8, [256, 256, 1024])
    iden10 = identity_block(iden9, [256, 256, 1024])
    proj4 = projection_block(iden10, [512, 512, 2048])
    iden11 = identity_block(proj4, [512, 512, 2048])
    iden12 = identity_block(iden11, [512, 512, 2048])
    avgpool = K.layers.AveragePooling2D(
        pool_size=7, strides=1, padding='same')(iden12)
    softmax = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init)(avgpool)
    model = K.models.Model(inputs=X, outputs=softmax)
    return model
