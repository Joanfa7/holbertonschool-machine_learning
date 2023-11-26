#!/usr/bin/env python3
""" Performs a algorithm to a HMM """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    Observation: numpy.ndarray of shape (T,) that contains the
    index of the observation
        T: number of observations
    Emission: numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
        Emission[i, j]: probability of observing j given the hidden state i
        N: number of hidden states
        M: number of all possible observations
    """
    # if not isinstance(Observation, np.ndarray) or len(Observation) == 0:
    #     return None, None
    # if not isinstance(Emission, np.ndarray) or len(Emission) != 2:
    #     return None, None
    # if not isinstance(Transition, np.ndarray) or len(Transition) != 2:
    #     return None, None
    # if not isinstance(Initial, np.ndarray) or len(Initial) != 2:
    #     return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    beta = np.zeros((N, T))
    beta[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for n in range(N):
            beta[n, t] = np.sum(Transition[n, :],
                                * Emission[:, Observation[t + 1]]
                                * beta[:, t + 1])

    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * beta[:, 0])
    return P, beta
