#!/usr/bin/env python3
""" Posterior Probability """

from scipy import special


def posterior(x, n, p1, p2):
    """
    Function that calculates the posterior probability for the various
    hypothetical probabilities of developing severe side effects given the
    data
    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        p1: lower bound on the range
        p2: upper bound on the range
    Returns: 1D numpy.ndarray containing the intersection of obtaining x and n
             with each probability in P, respectively
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal "
                         "to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise TypeError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise TypeError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    return special.btdtr(x + 1, n - x + 1, p1) - special.btdtr(x + 1,
                                                               n - x + 1, p2)