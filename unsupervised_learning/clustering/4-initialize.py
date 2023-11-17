#!/usr/bin/env python3
""" Initialize GMM """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


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
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0 or X.shape[0] < k:
        return None, None, None
    # Initializes the pi array with values equal to 1/k for each cluster.
    pi = np.full((k,), 1 / k)
    # Uses the kmeans function to initialize the m array with centroid 
    # means for each cluster. Also obtains the clss array that represents
    # the cluster assignments for each data point.
    m, clss = kmeans(X, k)
    # Initializes the S array with shape (k, d, d), where d is the dimension
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
