#!/usr/bin/env python3
""" Determinant """


def determinant(matrix):
    # Check if matrix is a list of lists
    if not (
        isinstance(
            matrix,
            list) and all(
            isinstance(
                row,
                list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    rows = len(matrix)
    if not all(len(row) == rows for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 0x0 matrix
    if rows == 0:
        return 1  # The determinant of a 0x0 matrix is defined as 1

    # Base case for 1x1 matrix
    if rows == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for larger matrices
    det = 0
    for col in range(rows):
        sub_matrix = [row[:col] + row[col + 1:]
                      for row in (matrix[:0] + matrix[0 + 1:])]
        det += ((-1)**col) * matrix[0][col] * determinant(sub_matrix)
    return det
