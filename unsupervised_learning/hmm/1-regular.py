#!/usr/bin/env python3
""" Regular Chains """

import numpy as np


def regular(P):
    """ determines the steady state probabilities of a regular markov chain
        P: square 2D np.ndarray (n, n) transition matrix
          P[i, j]: probability of transitioning from state i to state j
          n: number of states in the markov chain
          Returns: np.ndarray (1, n) containing the steady state probabilities,
        """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, n1 = P.shape
    if n != n1:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None
    if np.any(P <= 0):
        return None
    # Calculate the eigenvalues of eignvertors of P.T
    evals, evecs = np.linalg.eig(P.T)

    # Selects and normalizes the eigenvector corresponding to eigenvalue 1
    evecs = evecs[:, np.isclose(evals, 1)]
    evecs = evecs / evecs.sum()

    # Returns the stationary state of P
    return evecs.T
