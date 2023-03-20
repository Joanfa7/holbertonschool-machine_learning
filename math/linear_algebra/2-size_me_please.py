#!/usr/bin/env python3
"""Function calculates the shape of a matrix"""


def matrix_shape(matrix):
    
    matrix_lenth = len(matrix);

    for idx in range(matrix_lenth):
        return matrix_lenth[idx][0]