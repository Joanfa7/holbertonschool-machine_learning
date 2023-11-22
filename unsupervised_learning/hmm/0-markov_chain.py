#!/usr/bin/env python3
""" Markov Chain """

import numpy as np


def markov_chain(P, s, t=1):
    """ determines the probability of a markov chain being in a particular
        state after a specified number of iterations
        P: square 2D np.ndarray (n, n) transition matrix
          P[i, j]: probability of transitioning from state i to state j
          n: number of states in the markov chain
        s: np.ndarray (1, n) probability of starting in each state
        t: number of iterations to perform
        Returns: np.ndarray (1, n) probability of being in a specific state
          after t iterations, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, n2 = P.shape
    if n != n2:
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if type(t) is not int or t < 1:
        return None
    # Calculate the t-th power of matrix P
    Pk = np.linalg.matrix_power(P, t)

    # Calculate the state probabilities after t iterations
    state_probability = np.matmul(s, Pk)

    # Return the state probabilities after t iterations
    return state_probability
