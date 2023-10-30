#!/usr/bin/env python3

""" Function that calculates the minor matrix of a matrix """

determinant = __import__('0-determinant').determinant


def submatrix(matrix, i, j):
    """ Returns the submatrix of a matrix """
    sub = []
    for row in range(len(matrix)):
        if row != i:
            aux = []
            for col in range(len(matrix)):
                if col != j:
                    aux.append(matrix[row][col])
            sub.append(aux)
    return sub


def minor(matrix):
    """ Calculates the minor matrix of a matrix """
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
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
    minor = []
    for i in range(len(matrix)):
        minor.append([])
        for j in range(len(matrix)):
            minor[i].append(0)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            minor[i][j] = determinant(submatrix(matrix, i, j))
    return minor
