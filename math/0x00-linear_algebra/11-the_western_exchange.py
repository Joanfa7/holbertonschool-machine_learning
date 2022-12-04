#!/usr/bin/env python3
""" function that transposes a matrix"""


def np_transpose(matrix):
    """ transpose a matrix """
    n_matrix = matrix.copy()
    t_n_matrix = n_matrix.transpose()
    return t_n_matrix
