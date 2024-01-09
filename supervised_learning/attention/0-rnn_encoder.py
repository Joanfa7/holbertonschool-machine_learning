#!/usr/bin/env python3
"""Module that creates an encoder block for a transformer"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ A class representing an RNN Encoder.

    This class implements an RNN Encoder using GRU
    (Gated Recurrent Unit) cells.

    Args:
        vocab (int): The size of the vocabulary.
        embedding (int): The dimensionality of the embedding.
        units (int): The number of units in the GRU layer.
        batch (int): The batch size.

    Attributes:
        batch (int): The batch size.
        units (int): The number of units in the GRU layer.
        embedding (tf.keras.layers.Embedding): The embedding layer.
        gru (tf.keras.layers.GRU): The GRU layer.

    Methods:
        initialize_hidden_state: Initializes the hidden state of the GRU layer.
        call: Performs a forward pass through the encoder.

     """

    def __init__(self, vocab, embedding, units, batch):
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        ''' Initializes the hidden state of the GRU layer.

        Returns:
            tf.Tensor: The initial hidden state.

        '''
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        ''' Performs a forward pass through the encoder.

        Args:
            x (tf.Tensor): The input tensor.
            initial (tf.Tensor): The initial hidden state.

        Returns:
            tf.Tensor: The output tensor.
            tf.Tensor: The final hidden state.

        '''
        return self.gru(self.embedding(x), initial_state=initial)
