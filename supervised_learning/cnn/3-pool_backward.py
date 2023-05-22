#!/usr/bin/env python3
''' backward propagation over a pooling layer of a neural network'''

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    ''' backward propagation over a pooling layer of a neural network
        dA: numpy.ndarray (m, h_new, w_new, c_new) with partial derivatives
            m: number of examples
            h_new: height of output
            w_new: width of output
            c_new: number of channels
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) output of prev layer
            m: number of examples
            h_prev: height of prev layer
            w_prev: width of prev layer
            c_prev: number of channels in prev layer
        kernel_shape: tuple (kh, kw) size of kernel for pooling
            kh: kernel height
            kw: kernel width
        stride: tuple (sh, sw) strides for pooling
            sh: stride height
            sw: stride width
        mode: string 'max' or 'avg' for max or avg pooling
        Returns: partial derivatives with respect to previous layer (dA_prev)
    '''

    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    if mode == 'max':
                        A_slice = A_prev[i, h * sh:h * sh + kh,
                                         w * sw:w * sw + kw, c]
                        mask = (A_slice == np.max(A_slice))
                        dA_prev[i, h * sh:h * sh + kh,
                                w * sw:w * sw + kw, c] += (
                                    dA[i, h, w, c] * mask)
                    elif mode == 'avg':
                        dA_prev[i, h * sh:h * sh + kh,
                                w * sw:w * sw + kw, c] += (
                                    dA[i, h, w, c] / (kh * kw))
    return dA_prev
