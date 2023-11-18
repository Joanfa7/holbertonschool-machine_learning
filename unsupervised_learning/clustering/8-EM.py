#!/usr/bin/env python3
""" Expectation Maximization """

import numpy as np
inistialize = __import__('4-initialize').initialize
maximization = __import__('7-maximization').maximization
expectation = __import__('6-expectation').expectation


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ performs the expectation maximization for a GMM
        X: np.ndarray (n, d) data set
        k: positive int, number of clusters
        iterations: positive int, max number of iterations
        tol: non-negative float, tolerance of log likelihood
        verbose: bool, if True print log liklihood every 10 iterations
        Returns: pi, m, S, g, l, or None, None, None, None, None on failure
            pi: np.ndarray (k,), priors for each cluster
            m: np.ndarray (k, d), centroid means for each cluster
            S: np.ndarray (k, d, d), cov matrices for each cluster
            g: np.ndarray (k, n), probabilities for each data point
            l: log likelihood of the model
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k < 1:
        return None, None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = inistialize(X, k)
    g, ll = expectation(X, pi, m, S)
    prev = ll
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(i, ll.round(5)))
        pi, m, S = maximization(X, g)
        g, ll = expectation(X, pi, m, S)
        if abs(prev - ll) <= tol:
            break
        prev = ll
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(i + 1, ll.round(5)))
    return pi, m, S, g, ll