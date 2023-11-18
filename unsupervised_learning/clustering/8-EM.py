#!/usr/bin/env python3
""" Expectation Maximization """

import numpy as np
maximization = __import__('7-maximization').maximization
expectation = __import__('6-expectation').expectation


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ Expectation Maximization """
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
    n, d = X.shape
    pi = np.full((k,), 1/k)
    m = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(k, d))
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))
    g, like = expectation(X, pi, m, S)
    prev_like = 0
    for i in range(iterations):
        if verbose and (i % 10 == 0):
            print('Log Likelihood after {} iterations: {}'.format(i, like.round(5)))
        pi, m, S = maximization(X, g)
        g, like = expectation(X, pi, m, S)
        if abs(like - prev_like) <= tol:
            break
        prev_like = like
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(i + 1, like.round(5)))
    return pi, m, S, g, like