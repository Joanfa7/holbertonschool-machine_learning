#!/usr/bin/env python3
""" Module to create a deep RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ Function that performs forward propagation for a deep RNN"""
    t, m, i = X.shape
    _, _, h = h_0.shape
    time_step = range(t)
    H = np.zeros((t + 1, len(rnn_cells), m, h))
    H[0, :, :, :] = h_0
    Y = []
    for t in time_step:
        for layer in range(len(rnn_cells)):
            if layer == 0:
                h_prev = X[t]
            else:
                h_prev = H[t, layer - 1]
            h_next, y = rnn_cells[layer].forward(H[t, layer], h_prev)
            H[t + 1, layer] = h_next
        Y.append(y)
    Y = np.array(Y)
    return H, Y
