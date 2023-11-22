#!/usr/bin/env python3
""" Absorbing Chains """
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing
    P: np arr (n, n) transition matrix
      P[i, j]: prob of going from i to j
      n: number of states in the markov chain
    Returns: True if it is absorbing, False on failure
    """
    n, m = P.shape
    if n != m:
        return False

    # Identify absorbing states
    is_absorbing = np.diag(P) == 1
    if not np.any(is_absorbing):
        return False

    # Check if non-absorbing states can reach absorbing states
    non_absorbing_indives = np.where(is_absorbing == False)[0]
    if non_absorbing_indives.size == 0:
        return True

    Q = P[non_absorbing_indives, :][:, non_absorbing_indives]
    R = P[non_absorbing_indives, :][:, is_absorbing]

    # Calculate the fundamental matrix N = (I - Q)^-1
    try:
        N = np.linalg.inv(np.eye(len(non_absorbing_indives)) - Q)
    except np.linalg.LinAlgError:
        return False

    # Check if there's non-zero probability to reach an absorbing state
    B = np.dot(N, R)
    if np.any(B > 0):
        return False

    return False
