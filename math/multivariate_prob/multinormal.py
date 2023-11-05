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

    def pdf(self, x):
        ''' Calculates the PDF at a data point '''

        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)
        x_m = x - self.mean
        pdf = 1. / np.sqrt(((2 * np.pi) ** d) * cov_det) * \
            np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2)
        return pdf[0][0]
