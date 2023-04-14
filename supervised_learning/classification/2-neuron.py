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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

        @property
        def W(self):
            """ getter for W """
            return self.__W

        @property
        def b(self):
            """ getter for b """
            return self.__b

        @property
        def A(self):
            """ getter for A """
            return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron.
                Args:
                    X (numpy.ndarray): The input data with shape (nx, m).
                Returns:
                    The private attribute __A.
        """
        Z = np.matmul(self.W, X) + self.b
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
