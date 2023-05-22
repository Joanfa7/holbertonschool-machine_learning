#!/usr/bin/env python3
''' Convolutional Back Propagation '''

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    ''' Function that performs back propagation over a convolutional layer
        of a neural network '''
    m, hprev, wprev, cprev = A_prev.shape
    kh, kw, cprev, cnew = W.shape
    m, hnew, wnew, cnew = dZ.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0
    else:
        ph = int(((hprev - 1) * sh + kh - hprev) / 2) + 1
        pw = int(((wprev - 1) * sw + kw - wprev) / 2) + 1

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant', constant_values=0)

    dA_prev = np.zeros(A_prev_pad.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(m):
        for h in range(hnew):
            for w in range(wnew):
                for c in range(cnew):
                    start_h, end_h = h * sh, h * sh + kh
                    start_w, end_w = w * sw, w * sw + kw
                    dA_prev[i, start_h:end_h, start_w:end_w, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )
                    dW[:, :, :, c] += (
                        A_prev_pad[i, start_h:end_h, start_w:end_w, :] *
                        dZ[i, h, w, c]
                    )
                    db[:, :, :, c] += dZ[i, h, w, c]

    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    if padding == 'valid':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
