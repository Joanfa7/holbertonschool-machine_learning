#!/usr/bin/env python3
""" Class LSTMCell """
import numpy as np


class LSTMCell:
    """ Represent an LSTM unit """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """
        # Weights
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """ perform forward propagation """
        # Concat h_prev and x_t to match Wh dimensions
        x = np.concatenate((h_prev, x_t), axis=1)
        # Forget gate
        f = self.sigmoid(np.matmul(x, self.Wf) + self.bf)
        # Update gate
        u = self.sigmoid(np.matmul(x, self.Wu) + self.bu)
        # Candidate
        c = np.tanh(np.matmul(x, self.Wc) + self.bc)
        # Cell state
        c_next = f * c_prev + u * c
        # Output gate
        o = self.sigmoid(np.matmul(x, self.Wo) + self.bo)
        # Hidden state
        h_next = o * np.tanh(c_next)
        # Softmax activation function
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y

    def sigmoid(self, x):
        """ Sigmoid function """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """ Softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
