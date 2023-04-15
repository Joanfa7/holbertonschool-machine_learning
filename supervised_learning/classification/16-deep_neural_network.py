#!/usr/bin/env python3
""" DeepNeuralNetwork """

import numpy as np


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork """

    def __init__(self, nx, layers):
        """ Constructor """
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights["W{}".format(
                i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]
