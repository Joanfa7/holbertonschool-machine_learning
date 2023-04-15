#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork """

    def __init__(self, nx, layers):
        """ Class constructor """
        if isinstance(layers, int):
            layers = [layers]

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        """ Getter method for L """
        return self.__L

    @property
    def cache(self):
        """ Getter method for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter method for weights """
        return self.__weights
