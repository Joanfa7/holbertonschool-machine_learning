#!/usr/bin/env python3
""" Neuron class. """

import numpy as np

class Neuron:
    """ Neuron class.""" 
    def __init__(self, nx):
        """ Class constructor. """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0