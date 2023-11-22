#!/usr/bin/env python3
""" Forward Algorithm """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model
    Observation: np arr (T,) with index of observation(s)
      T: number of observations
    Emission: np arr (N, M) of emmission probabilities
      N: number of hidden states
      M: number of possible observations
    Transition: np arr (N, N) of transition probabilities
    Initial: np arr (N, 1) of prob of starting in specific hidden state
    Returns: P, F, or None, None on failure
      P: likelihood of the observations given the model
      F: np arr (N, T) of forward path probabilities
        F[i j]: prob of being in hidden state i at time j given the
                previous observations
    """
    if (
        not isinstance(Observation, np.ndarray)
        or not isinstance(Emission, np.ndarray)
        or not isinstance(Transition, np.ndarray)
        or not isinstance(Initial, np.ndarray)
    ):
        return None, None

    # Number of observations (T) and hidden states (N)
    T = Observation.shape[0]
    N = Emission.shape[0]

    # Initialize the forward path probabilities F[i j]
    F = np.zeros((N, T))
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])

    # Iterate through each time step
    for t in range(1, T):
        for n in range(N):
            F[n, t] = (
                np.sum(np.multiply(F[:, t - 1], Transition[:, n]))
                * Emission[n, Observation[t]]
            )

    # Calculate the total probability of the observation
    P = np.sum(F[:, T - 1])

    return P, F
