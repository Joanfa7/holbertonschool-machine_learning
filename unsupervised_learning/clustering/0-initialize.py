#!/usr/bin/env python3
""" initializes cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means
        X: np.ndarray (n, d) dataset to cluster
          n: number of data points
          d: number of dimensions
        k: positive int, number of clusters
        Returns: np.ndarray (k, d) initialized centroids for each cluster,"""
    # Check if X is a numpy ndarray with 2 dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Check if k is a positive integer
    if not isinstance(k, int) or k <= 0:
        return None

    # Get the number of data points and dimensions
    n, d = X.shape

    # Calculate the minimum and maximum values of X along each dimension
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)

    # Generate a numpy ndarray of random values between the minimum and
    # maximum values of X
    centroids = np.random.uniform(low, high, (k, d))
    return centroids
