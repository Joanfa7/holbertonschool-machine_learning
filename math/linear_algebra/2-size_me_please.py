#!/usr/bin/env python3
"""Function calculates the shape of a matrix"""


def matrix_shape(matrix):
    """ returns the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape