#!/usr/bin/env python3
""" Method that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """ Method that calculates the derivative of a polinomial"""
    if not isinstance(poly, list):
        return None
    elif len(poly) == 1:
        return [0]
    elif len(poly) == 0:
        return None
    else:
        derivative = [poly[i] * i for i in range(1, len(poly))]
        return derivative
