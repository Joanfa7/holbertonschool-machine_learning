#!/usr/bin/env python3
""" Method that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """ Method that calculates the derivative of a polinomial"""
    if not isinstance(poly, list):
        return None
    derivative = [poly[i] * i for i in range(1, len(poly))]
    if derivative == [0]:
        return derivative
    else:
        return derivative
