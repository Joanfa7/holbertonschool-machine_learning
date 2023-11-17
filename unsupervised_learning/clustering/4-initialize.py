#!/usr/bin/env python3
""" Initialize GMM """

import numpy as np


def initialize(X, k):
    """ initializes variables  a Gaussian Mixture Model
        X: np.ndarray (n, d) of data set
          n: number of data points
          d: number of dimensions
        k: positive int of number of clusters
        Returns: pi, m, S, or None, None, None on failure
          pi: np.ndarray (k,) of priors  each cluster
          m: np.ndarray (k, d) of centroid means  each cluster
          S: np.ndarray (k, d, d) of covariance matrices each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    # Get the number of data points (n) and dimensions (d) from the shape of X
    n, d = X.shape
    # Initialize the priors pi with equal probabilities each cluster
    pi = np.full((k,), 1 / k)
    # Initialize the centroid means m with random values within the range of X
    # Initialize the centroid means m with random values within the range of X
    m = np.random.uniform(X.min(axis=0), X.max(axis=0), size=(k, d))
    # Initialize the covariance matrices S as identity matrices with the same
    # dimension as X
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))
    return pi, m, S
