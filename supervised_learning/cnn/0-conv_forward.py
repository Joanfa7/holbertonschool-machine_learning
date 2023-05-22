#!/usr/bin/env python3
''' Convolutional Forward Propagation '''

import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1,1)):
    ''' performs forward propagation over a convolutional layer of a neural
        network
        @A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) with the
                 output of the previous layer
            @m: number of examples
            @h_prev: height of the previous layer
            @w_prev: width of the previous layer
            @c_prev: number of channels in the previous layer
        @W: numpy.ndarray of shape (kh, kw, c_prev, c_new) with the kernels for
            the convolution
            @kh: filter height
            @kw: filter width
            @c_prev: number of channels in the previous layer
            @c_new: number of channels in the output
        @b: numpy.ndarray of shape (1, 1, 1, c_new) with the biases applied to
            the convolution
        @activation: activation function applied to the convolution
        @padding: string that is either same or valid, indicating the type of
                  padding used
        @stride: tuple of (sh, sw) containing the strides for the convolution
            @sh: stride for the height
            @sw: stride for the width
        Retruns: output of the convolutional layer
    '''
    # Retrieving dimensions from A_prev shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieving dimensions from W's shape
    (kh, kw, c_prev, c_new) = W.shape

    # Retrieving information from 'stride'
    sh, sw = stride

    # Padding
    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0

    # Applying padding to the previous layer
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant', constant_values=0)
    
    # Computing the dimensions of the CONV output volume
    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1

    # Initializing the output volume Z with zeros
    Z = np.zeros((m, h_new, w_new, c_new))

    # Looping over the vertical(h) and horizontal(w) axis of the output volume
    for i in range(h_new):
        for j in range(w_new):
            # Looping over the channels(c_new)
            for k in range(c_new):
                # Creating slices
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw

                # Computing convolution
                Z[:, i, j, k] = np.sum(np.multiply(
                    A_prev_pad[:, v_start:v_end, h_start:h_end, :],
                    W[:, :, :, k]), axis=(1, 2, 3))
                
                # Adding bias
                Z[:, i, j, k] = Z[:, i, j, k] + b[:, :, :, k].reshape(1, 1, 1)

    # Applying activation
    A = activation(Z)

    return A

