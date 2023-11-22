#!/usr/bin/env python3
""" Viterbi """

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ calculates the most likely sequence of hidden states for a
        hidden markov model
        Observation: np arr (T,) index of observation(s)
        Emission: np arr (N, M) emission prob of specific
                  observation given hidden state
        Transition: np arr (N, N) transition probabilities
        Initial: np arr (N, 1) prob of starting in specific hidden state
        Returns: path, P, or None, None on failure
                 path: np arr (best path) of hidden states
                 P: prob of obtaining path sequence """

    # type and len check
    if not isinstance(Observation, np.ndarray) or \
            len(Observation.shape) is not 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or \
            len(Emission.shape) is not 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or \
            len(Transition.shape) is not 2 or \
            Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or \
            len(Initial.shape) is not 2 or \
            Initial.shape[1] is not 1:
        return None, None

    # var init
    T = Observation.shape[0]
    N, M = Emission.shape
    path = []
    P = 0

    # base case
    F = np.zeros((N, T))
    B = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])

    # recursion
    for t in range(1, T):
        for s in range(N):
            F[s, t] = np.max(np.multiply(F[:, t - 1],
                                         Transition[:, s])) * \
                Emission[s, Observation[t]]
            B[s, t] = np.argmax(np.multiply(F[:, t - 1],
                                            Transition[:, s]))

    # path init
    path.insert(0, np.argmax(F[:, T - 1]))
    P = np.max(F[:, T - 1])

    # path loop
    for t in range(T - 2, -1, -1):
        path.insert(0, int(B[path[0], t]))

    return path, P
