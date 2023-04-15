#!/usr/bin/env python3
""" DeepNeuralNetwork """

import numpy as np


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork """

    def __init__(self, nx, layers):
        """ Class constructor """
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights["W{}".format(i + 1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)
            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]
