#!/usr/bin/env python3
""" likelihood function """

import numpy as np


def likelihood(x, n, p):
    """ liklihood function """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p, np.ndarray):
        raise TypeError("p must be a 1D numpy.ndarray")
    if len(p.shape) != 1:
        raise TypeError("p must be a 1D numpy.ndarray")
    for i in p:
        if i < 0 or i > 1:
            raise ValueError("All values in p must be in the range [0, 1]")
    fact = np.math.factorial
    tmp = fact(n) / (fact(x) * fact(n - x))
    return tmp * (p ** x) * ((1 - p) ** (n - x))
