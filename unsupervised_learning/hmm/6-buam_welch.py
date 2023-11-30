#!/usr/bin/env python3
""" Function that performs the Baum-Welch algorithm for a hidden markov model"""

import numpy as np


def bau_welch(Observations, Transitions, Emission, Initial, iterations=1000):
    """ Function that performs the Baum-Welch algorithm for a hidden markov model"""
    # create the fuction
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transitions) is not np.ndarray or len(Transitions.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    T = Observations.shape
    N, M = Emission.shape
