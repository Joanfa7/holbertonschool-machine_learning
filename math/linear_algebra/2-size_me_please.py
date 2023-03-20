#!/usr/bin/env python3
"""Function calculates the shape of a matrix"""


def matrix_shape(matrix):
    row = len(matrix)
    col = len(matrix[0])
    shape = [row, col]
    return(shape)