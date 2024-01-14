#!/usr/bin/env python3
""" RNN Decoder """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    ''' RNNDecoder class '''
    def __init__(self, vocab, embedding, units, batch):
        ''' Initializer Constructor '''
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.units = units
        self.batch = batch
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        ''' call method '''
        batch, units = s_prev.shape
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, hidden = self.gru(x)
        outputs = tf.reshape(outputs,  [batch, -1])
        y = self.F(outputs)
        return y, hidden
