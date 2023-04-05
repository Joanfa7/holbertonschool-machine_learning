#!/usr/bin/env python3
""" Exponential distribution class """


class Exponential:
    """ Exponential distribution class"""

    def __init__(self, data=None, lambtha=1):
        ''' Constructor
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        '''
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        ''' Calculates the value of the PDF for a given time period
            x: time period
        '''
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)
