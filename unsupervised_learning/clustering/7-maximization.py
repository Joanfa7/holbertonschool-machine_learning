#!/usr/bin/env python3
""" Maximization """

import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm a GMM:

    -> X is a numpy.ndarray of shape (n, d) containing the data set
    -> g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities  each data point in each cluster
    -> Returns: pi, m, S, or None, None, None on failure
        * pi is a numpy.ndarray of shape (k,) containing the updated
            priors  each cluster
        * m is a numpy.ndarray of shape (k, d) containing the updated
            centroid means  each cluster
        * S is a numpy.ndarray of shape (k, d, d) containing the
            updated covariance matrices  each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    # Get the number of samples (n) and the number of features (d)
    n, d = X.shape
    # Get the number of clusters (k)
    k = g.shape[0]
    if not np.isclose(np.sum(g, axis=0), np.ones((n,))).all():
        return None, None, None
    # Calculate the new priors, means and covariances
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        # Calculate the updated priors each cluster
        pi[i] = np.sum(g[i]) / n
        # Calculate the updated means each cluster
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        # Calculate the updated covariances each cluster
        S[i] = np.matmul(g[i] * (X - m[i]).T, (X - m[i])) / np.sum(g[i])
    return pi, m, S
