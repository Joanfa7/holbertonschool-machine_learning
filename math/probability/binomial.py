#!/usr/bin/env python3
'''Create a binomial distribution'''


class Binomial:
    '''Create a binomial distribution'''

    def __init__(self, data=None, n=1, p=0.5):
        '''Class constructor'''
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.n = len(data)
            self.p = sum(data) / len(data)

    def pmf(self, k):
        '''Calculates the value of the PMF for a given number of successes'''
        if k < 0:
            return 0
        return (self.factorial(self.n) / (self.factorial(k) *
                                          self.factorial(self.n - k))) *\
            (self.p ** k) * ((1 - self.p) ** (self.n - k))
