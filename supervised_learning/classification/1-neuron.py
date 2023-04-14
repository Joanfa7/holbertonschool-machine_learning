#!/usr/bin/env python3
""" Neuron class. """

import numpy as np


class Neuron:
    """ Neuron class."""

    def __init__(self, nx):
        """ Class constructor.
                Args: nx (int): number of input features to the neuron.
                Attributes:
                    W (numpy.ndarray): The weights vector for the neuron.
                    b (int): The bias for the neuron.
                    A (int): The activated output of the neuron (prediction).
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self._W = np.random.randn(nx).reshape(1, nx)
        self._b = 0
        self._A = 0

        @property
        def W(self):
            """ getter for W """
            return self._W

        @property
        def b(self):
            """ getter for b """
            return self._b

        @property
        def A(self):
            """ getter for A """
            return self._A
