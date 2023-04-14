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
        
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0

        @property
        def get_W(self):
            """ getter for W """
            return self.__W

        @property
        def get_b(self):
            """ getter for b """
            return self.__b

        @property
        def get_A(self):
            """ getter for A """
            return self.__A


