#!/usr/bin/env python3
"""PCA module"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    u, Sigma, vh = np.linalg.svd(X, full_matrices=False)
    cum = np.cumsum(Sigma) / np.sum(Sigma)
    r = (np.argwhere(cum >= var))[0, 0]
    w = vh.T
    wr = w[:, :r +1]
    return wr