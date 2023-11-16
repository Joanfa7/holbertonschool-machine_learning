#!/usr/bin/env python3
""" Perform k-means clustering on the data. """

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """ Perform k-means clustering on the data.
        Args:
            X: (numpy.ndarray) containing the dataset.
            k: (int) containing the number of clusters.
            iterations: (int) containing the maximum number of iterations.
        Returns:
            C: (numpy.ndarray) containing the centroid means for each cluster.
            clss: (numpy.ndarray) containing the index of the cluster in C
                  that each data point belongs to.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    n, d = X.shape
    C = initialize(X, k)
    for i in range(iterations):
        C_copy = np.copy(C)
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        clss = np.argmin(distances, axis=0)
        for j in range(k):
            if len(X[clss == j]) == 0:
                C[j] = initialize(X, 1)[0]
            else:
                C[j] = np.mean(X[clss == j], axis=0)
        distances = np.linalg.norm(C_copy - C, axis=1)
        if np.all(distances == 0):
            break
    return C, clss