#!/usr/bin/env python3

""" cofactor functionv"""

determinant = __import__('0-determinant').determinant
submatrix = __import__('1-minor').submatrix


def cofactor(matrix):
    """ Function that calculates the cofactor matrix of a matrix """
    if not isinstance(
            matrix,
            list) or len(matrix) == 0 or not isinstance(
            matrix[0],
            list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    if len(matrix) == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]
    cofactor = []
    for i in range(len(matrix)):
        cofactor.append([])
        for j in range(len(matrix)):
            cofactor[i].append(0)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cofactor[i][j] = ((-1)**(i + j)) * \
                determinant(submatrix(matrix, i, j))
    return cofactor
