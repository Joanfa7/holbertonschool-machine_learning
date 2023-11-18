#!/usr/bin/env python3
""" Expectation """

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorith a GMM:

    -> X is a numpy.ndarray of shape (n, d) containing the data set
    -> pi is a numpy.ndarray of shape (k,) containing the priors  each cluster
    -> m is a numpy.ndarray of shape (k, d) containing the centroid means
        each cluster
    -> S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices
        each cluster
    -> Returns: g, l, or None, None on failure
        * g is a numpy.ndarray of shape (k, n) containing the posterior probabilities
             each data point in each cluster
        * l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        # Check if the dimensions of m, S, and pi are consistent with d
        return None, None
    if k != m.shape[0] or k != S.shape[0] or k != pi.shape[0]:
        # Check if the dimensions of m, S, and pi are consistent with k
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        # Check if the sum of pi is approximately equal to 1
        return None, None
    g = np.zeros((k, n))
    for i in range(k):
        P = pdf(X, m[i], S[i])
        # Calculate the probability density function P  each cluster
        g[i] = pi[i] * P
    g = g / np.sum(g, axis=0)
    # Normalize the posterior probabilities g
    l = np.sum(np.log(np.sum(g, axis=0)))
    # Calculate the total log likelihood l
    return g, l
