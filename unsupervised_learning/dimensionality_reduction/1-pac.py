#!/usr/bin/env python3
"""PCA module"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    x_mean = X - np.mean(X, axis=0)
    u, Sigma, vh = np.linalg.svd(x_mean)
    w = vh.T
    Wr = w[:,ndim]
    T = np.dot(x_mean, Wr)
    return T
