#!/usr/bin/env python3
""" Function that returns the sum of i squared"""


def summation_i_squared(n):
    ''' Function that returns the sum of i squared '''
    if not isinstance(n, int) or n < 1:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
