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

        dist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)\
            - (2 * np.dot(X1, X2.T))


        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist)

    def predict(self, X_s):
        """ predicts the mean and standard deviation of points
            in a Gaussian process """
        X_s = np.array(X_s)
        print(f"X_s.shape: {X_s.shape}")
        K = self.K
        print(f"K.shape: {K.shape}")
        K_s = self.kernel(self.X, X_s)
        print(f"K_s.shape: {K_s.shape}")
        K_ss = self.kernel(X_s, X_s)
        print(f"K_ss.shape: {K_ss.shape}")
        K_inv = np.linalg.inv(K)
        print(f"K_inv.shape: {K_inv.shape}")

        mu_s = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        print(f"mu_s.shape: {mu_s.shape}")
        sigma = K_ss - K_s.T.dot(K_inv).dot(K_s)
        print(f"sigma.shape: {sigma.shape}")
        sigma = np.diag(sigma)
        print(f"sigma.shape: {sigma.shape}")

        return mu_s, sigma