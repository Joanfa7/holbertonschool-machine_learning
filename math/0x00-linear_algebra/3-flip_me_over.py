#!/usr/bin/env python3
"""Function that returns transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """ Function that transpose a 2D matrix"""
    if not matrix:
        return []
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
