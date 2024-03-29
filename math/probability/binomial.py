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
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            self.p = 1 - variance / mean
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        '''Calculates the value of the PMF for a given number of successes'''
        if k < 0:
            return 0
        k = int(k)
        if k > self.n:
            return 0
        else:
            return (self.n * self.p) ** k * (1 - self.p) ** (self.n - k)

    def cdf(self, k):
        '''Calculates the value of the CDF for a given number of successes'''
        if k < 0:
            return 0
        k = int(k)
        if k > self.n:
            return 1
        else:
            return sum([self.pmf(i) for i in range(k + 1)])
