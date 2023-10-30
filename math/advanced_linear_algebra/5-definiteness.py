

import numpy as np


def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None  # Return None if matrix is not square

    # Ensure the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    positive_eigenvalues = np.all(eigenvalues > 0)
    non_negative_eigenvalues = np.all(eigenvalues >= 0)
    negative_eigenvalues = np.all(eigenvalues < 0)
    non_positive_eigenvalues = np.all(eigenvalues <= 0)

    if positive_eigenvalues:
        return 'Positive definite'
    elif non_negative_eigenvalues and not positive_eigenvalues:
        return 'Positive semi-definite'
    elif negative_eigenvalues:
        return 'Negative definite'
    elif non_positive_eigenvalues and not negative_eigenvalues:
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
