#!/usr/bin/env python3
""" 0. Mean and Covariance """

import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set
        X: np.ndarray (n, d) dataset
          n: number of data points
          d: number of dimensions in each data point
        Returns: mean, cov
          mean: np.ndarray (1, d) mean of the data set
          cov: np.ndarray (d, d) covariance matrix of the data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)
    return mean, cov
