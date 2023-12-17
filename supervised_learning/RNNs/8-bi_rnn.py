#!/usr/bin/env python3
""" RNNs """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ Perform forward propagation for a bidirectional RNN """
    t, m, i = X.shape
    _, h = h_0.shape
    time_steps = range(t)
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    for time_step in time_steps:
        if time_step == 0:
            Hf[time_step] = bi_cell.forward(h_0, X[time_step])
            Hb[t - 1] = bi_cell.backward(h_t, X[t - 1])
        else:
            Hf[time_step] = bi_cell.forward(Hf[time_step - 1], X[time_step])
            Hb[t - time_step - 1] = bi_cell.backward(Hb[t - time_step],
                                                     X[t - time_step - 1])
    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
