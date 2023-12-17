#!/usr/bin/env python3
""" Class BidirectionalCell that represents a bidirectional cell of an RNN """
import numpy as np


class BidirectionalCell:
    """ Class that represents a bidirectional cell of an RNN """
    def __init__(self, i, h, o):
        """ Constructor """
        # Weights
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h + h, o)
        # Biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ Calculate the hidden state """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)
        return h_next