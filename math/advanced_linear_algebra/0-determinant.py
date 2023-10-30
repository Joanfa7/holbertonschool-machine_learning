#!/usr/bin/env python3
""" Determinant """

def determinant(matrix):
    """ Calculates the determinant of a matrix
        matrix: list of lists whose determinant should be calculated
        Returns: the determinant of matrix
    """
    if type(matrix) is not list or matrix == []:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - \
               matrix[0][1] * matrix[1][0]
    det = 0
    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        new_m = [[row[n] for n in range(len(matrix)) if n != i]
                 for row in rows]
        det += k * (-1) ** i * determinant(new_m)
    return det