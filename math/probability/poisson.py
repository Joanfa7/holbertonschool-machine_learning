#!/usr/bin/env python3
""" Poisson distribution class """


class Poisson:
    """ Poisson distribution class"""

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
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes
            k: number of successes
        """
        if k < 0:
            return 0
        k = int(k)
        fact = 1
        for i in range(1, k + 1):
            fact *= i
        return (self.lambtha ** k * 2.7182818285 ** (-self.lambtha)) / fact
