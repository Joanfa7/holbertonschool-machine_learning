#!/usr/bin/env python3
"""Module that creates an encoder block for a transformer"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ Class RNNEncoder """

    def __init__(self, vocab, embedding, units, batch):
        ''' '''
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        ''' '''
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        ''' '''
        return self.gru(self.embedding(x), initial_state=initial)