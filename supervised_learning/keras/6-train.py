#!/usr/bin/env python3
''' Function that trains a model using mini-batch gradient descent '''

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                verbose=True, early_stopping=False, patience=0, shuffle=False):
    ''' Function that trains a model using mini-batch gradient descent '''
    callbacks = []
    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
