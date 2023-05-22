#!/usr/bin/env python3
''' Convolutional Forward Propagation '''

import numpy as np


def conv_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    ''' performs forward propagation over a pooling layer of a neural network
        A_prev ndarray (m, h_prev, w_prev, c_prev) input to pool layer
            m is number of examples
            h_prev is height of prev layer
            w_prev is width of prev layer
            c_prev is number of channels in prev layer
        kernel_shape is tuple (kh, kw) size of kernel for pooling
            kh is kernel height
            kw is kernel width
        stride is tuple (sh, sw) of strides for pooling
            sh is stride for height
            sw is stride for width
        mode is string either max or avg to indicate type of pooling
        Returns: output of pooling layer
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = int(((h_prev - kh) / sh) + 1)
    w_out = int(((w_prev - kw) / sw) + 1)
    A = np.zeros((m, h_out, w_out, c_prev))
    for i in range(h_out):
        for j in range(w_out):
            if mode == 'max':
                A[:, i, j, :] = np.max(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :], axis=(1, 2))
            if mode == 'avg':
                A[:, i, j, :] = np.mean(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :], axis=(1, 2))
    return A