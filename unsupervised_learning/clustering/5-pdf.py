#!/usr/bin/env python3
""" calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """ calculates the probability density function
    of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    n, d = X.shape
    n, d = X.shape

    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        # Checks if the dimensions of m and S are compatible with the
        # dimensions of X.
        return None

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    den = np.sqrt((2 * np.pi) ** d * det)

    # Calculates the squared Mahalanobis distance between each data point and
    # the mean vector.
    fac = np.einsum('...k,kl,...l->...', X - m, inv, X - m)

    # Calculates the probability density function  each data point.
    pdf = np.exp(-fac / 2) / den

    # Sets a minimum value to avoid division by zero.
    pdf = np.maximum(pdf, 1e-300)
    return np.maximum(pdf, 1e-300)
