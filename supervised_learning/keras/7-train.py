#!/usr/bin/env python3
''' Function that trains a model using mini-batch gradient descent '''

import tensorflow.keras as K


def train_model(network, data, labels, batch_zize, epochs, validation_data=None,
                early_stopping=False, patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    ''' Function that trains a model using mini-batch gradient descent '''
    callbacks = []
    if validation_data:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(patience=patience))
        if learning_rate_decay:
            def scheduler(epoch):
                ''' Scheduler function '''
                return alpha / (1 + decay_rate * epoch)
            callbacks.append(K.callbacks.LearningRateScheduler(scheduler,
                                                               verbose=1))
    return network.fit(data, labels, batch_size=batch_zize, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
