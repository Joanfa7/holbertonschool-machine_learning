#!/usr/bin/env python3
""" optimal number of clusters """

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance
        X: np.ndarray (n, d) of data set
          n: number of data points
          d: number of dimensions
        kmin: positive int of min number of clusters to check
        kmax: positive int of max number of clusters to check
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax is None:
        # Set kmax to the number of data points if not provided
        kmax = X.shape[0]
    results = []
    d_vars = []
    # Iterate over different cluster sizes
    for k in range(kmin, kmax + 1):
        # Perform K-means clustering using the kmeans function
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        # Calculate variance for the current cluster size
        if k == kmin:
            var_min = variance(X, C)
        var = variance(X, C)
        # Calculate the difference in variance from the smallest cluster size
        d_vars.append(var_min - var)
    return results, d_vars
