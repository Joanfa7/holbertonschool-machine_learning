#!/usr/bin/env python3
""" RNNs """
import numpy as np


class BidirectionalCell:
    """ Represents a bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """ Class constructors """
        # Weights
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        # Biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Calcuclate the hidden state in the forward direcitons
            for one time step
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh((h_x @ self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """ Calcuclate the hidden state in the backward direcitons
            for one time step
        """
        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh((h_x @ self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """ Calculate all outputs for the RNN """
        t, m, _ = H.shape
        time_steps = range(t)
        o = self.by.shape[1]
        Y = np.zeros((t, m, o))
        for time_step in time_steps:
            y_pred = self.softmax((H[time_step] @ self.Wy) + self.by)
            Y[time_step] = y_pred
        return Y

    def softmax(self, Y):
        """ Calculate the softmax """
        return np.exp(Y) / np.sum(np.exp(Y), axis=1, keepdims=True)
