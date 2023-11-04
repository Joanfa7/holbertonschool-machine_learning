#!/usr/bin/env python3
""" 1. Correlation """

import numpy as np


def correlation(C):
    """ calculates a correlation matrix
        C: np.ndarray (d, d) covariance matrix
          d: number of dimensions
        Returns: np.ndarray (d, d) correlation matrix
    """
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        raise TypeError("C must be a 2D numpy.ndarray")
    d1, d2 = C.shape
    if d1 != d2:
        raise ValueError("C must be a 2D square matrix")
    diag = np.diag(C)
    diag = np.sqrt(diag)
    outer = np.outer(diag, diag)
    correlation = C / outer
    return correlation
