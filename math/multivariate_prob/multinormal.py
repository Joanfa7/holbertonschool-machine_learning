#!/usr/bin/env python3
""" Multinormal """""
import numpy as np


class MultiNormal():
    """ Class Multinormal """

    def __init__(self, data):
        """ Class constructor """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        X = data - self.mean
        self.cov = np.dot(X, X.T) / (n - 1)
