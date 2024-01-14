#!/usr/bin/env python3
'''Self Attention Module '''

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    ''' Self Attention class '''
    def __init__(self, units):
        ''' Initializer Constructor '''
        super(SelfAttention, self).__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units=self.units)
        self.U = tf.keras.layers.Dense(units=self.units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        ''' call method '''
        s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(weights * hidden_states, axis=1)
        return context_vector, weights
