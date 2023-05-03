#!/usr/bin/env python3
""" Dropout Gradient Descent """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights of a neural network with Dropout regularization
        using gradient descent
        @Y: is a one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the data
            @classes: is the number of classes
            @m: is the number of data points
        @weights: is a dictionary of the weights and biases of the neural
                  network
        @cache: is a dictionary of the outputs and dropout masks of each
                layer of the neural network
        @alpha: is the learning rate
        @keep_prob: is the probability that a node will be kept
        @L: is the number of layers of the network
        All layers except the last should use the tanh activation function
        The last layer should use the softmax activation function
        Returns: Nothing. The weights of the network should be updated in place
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        dW = np.matmul(dZ, A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(W.T, dZ)
        if i > 1:
            dA = dA * (1 - A * A)
            dA = dA * cache["D" + str(i - 1)]
            dA = dA / keep_prob
        dZ = dA
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db
