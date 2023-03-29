#!/usr/bin/env python3

"""Function that calculates de integral of a polynomial"""


def poly_integral(poly, C=0):
    """Function that calculates de integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    elif len(poly) == 0:
        return [C] 
    elif len(poly) == 1:
        return [C, poly[0]]
    else:
        integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
        rounded_integral = [round(i, 2) for i in integral]
        return rounded_integral
