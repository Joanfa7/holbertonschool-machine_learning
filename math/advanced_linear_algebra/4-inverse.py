#!/usr/bin/env python3

"""Function def inverse(matrix) that calculates the inverse of a matrix"""


def inverse(matrix):
    if not isinstance(
        matrix,
        list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    # Check for 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None
        return [[1 / matrix[0][0]]]

    # Create an identity matrix
    identity = [[float(i == j) for j in range(n)] for i in range(n)]

    # Augment the matrix with the identity matrix
    augmented_matrix = [matrix[i] + identity[i] for i in range(n)]

    # Apply Gauss-Jordan Elimination
    for i in range(n):
        # Make the diagonal contain all 1's
        diag_value = augmented_matrix[i][i]
        if diag_value == 0:
            # Find a row below with a non-zero value in the same column
            for j in range(i + 1, n):
                if augmented_matrix[j][i] != 0:
                    augmented_matrix[i], augmented_matrix[j] = augmented_matrix[j], augmented_matrix[i]
                    diag_value = augmented_matrix[i][i]
                    break
            else:
                # Matrix is singular if no non-zero value is found
                return None

        # Scale the row to make the diagonal value 1
        augmented_matrix[i] = [x / diag_value for x in augmented_matrix[i]]

        # Zero out the other rows
        for j in range(n):
            if j != i:
                factor = -augmented_matrix[j][i]
                augmented_matrix[j] = [
                    augmented_matrix[j][k] +
                    factor *
                    augmented_matrix[i][k] for k in range(
                        2 *
                        n)]

    # Extract the inverse matrix from the augmented matrix
    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix
