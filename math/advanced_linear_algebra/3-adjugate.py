#!/usr/bin/env python3

""" adjugate function """

cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """ Function that calculates the adjugate matrix of a matrix """
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
    cofactor_matrix = cofactor(matrix)
    adjugate = []
    for i in range(len(matrix)):
        adjugate.append([])
        for j in range(len(matrix)):
            adjugate[i].append(0)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            adjugate[j][i] = cofactor_matrix[i][j]
    return adjugate
