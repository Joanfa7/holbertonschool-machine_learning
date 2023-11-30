#!/usr/bin/env python3
""" Initialize Gaussian Process """

import numpy as np


class GaussianProcess:
    """ Gaussian Process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Constructor """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ calculate the covariance kernel matrix """
        X1 = np.array(X1)
        X2 = np.array(X2)

        dist = np.sum(X1 ** 2, 1).reshape(-1, 1) +\
            np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

        sigma = 1.0

        return np.exp(-dist / (2 * sigma ** 2))
