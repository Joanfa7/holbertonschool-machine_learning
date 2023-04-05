#!/usr/bin/env python3
""" Poisson distribution class """

class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """ Constructor 
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))