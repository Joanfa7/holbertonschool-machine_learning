#!/usr/bin/env python3
'''Class RRNCell '''
import numpy as np


class RNNCell:
    """ Represent a cell of a simple RNN """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """
        # Weights
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_preg, x_t):
        """
        Public instance method
        Args:
            x_t: is a numpy.ndarray of shape (m, i) that contains
            the data input for the cell
                m is the batche size for the data
            h_prev: is a numpy.ndarray of shape (m, h) containing
            the previous hidden state
        Returns: h_next, y
        """
        # Concat h_prev and x_t to match Wh dimensions
        x = np.concatenate((h_preg, x_t), axis=1)
        h_next = np.tanh(np.dot(x, self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        # Softmax activation function
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y

    def sigmoid(self, x):
        """ Sigmoid function """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """ Softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
