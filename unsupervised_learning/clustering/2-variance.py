#!/usr/bin/env python3
""" Calculates the total intra-cluster variance for a data set """

import numpy as np


def variance(X, C):
    """ Calculate the variance of a dataset with respect to cluster centroids.

    Args:
        X: numpy.ndarray, shape (n, d), representing the dataset.
        C: numpy.ndarray, shape (k, d), representing the cluster centroids.

    Returns:
        var: float, representing the variance of
        the dataset with respect to the cluster centroids.
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    # Get the number of data points (n) and dimensions (d) in X
    n, d = X.shape
    # Get the number of clusters (k)
    k = C.shape[0]
    # Calculate the distances between each data point in X and each centroid
    # in C
    dist = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    # Find the minimum distance each data point
    clss = np.min(dist, axis=0)
    # Calculate the variance by summing the squares of the minimum distances
    var = np.sum(clss ** 2)
    return var
