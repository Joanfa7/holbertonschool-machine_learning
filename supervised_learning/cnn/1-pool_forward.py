#!/usr/bin/env python3
''' Pooling Forward Prop '''

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    ''' performs forward propagation over a pooling layer of a neural network
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
        Returns: output of pooling layer
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = ((h_prev - kh) // sh) + 1
    pw = ((w_prev - kw) // sw) + 1
    pooled = np.zeros((m, ph, pw, c_prev))
    for i in range(ph):
        for j in range(pw):
            if mode == 'max':
                pooled[:, i, j, :] = np.max(A_prev[:, i * sh:i * sh + kh,
                                                   j * sw:j * sw + kw, :],
                                            axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(A_prev[:, i * sh:i * sh + kh,
                                                    j * sw:j * sw + kw, :],
                                             axis=(1, 2))
    return pooled
