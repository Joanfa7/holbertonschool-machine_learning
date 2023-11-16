#!/usr/bin/env python3
""" Perform k-means clustering on the data. """

import numpy as np


def kmeans(X, k, iterations=1000):
    """ Perform k-means clustering on the data.
        Args:
            X: (numpy.ndarray) containing the dataset.
            k: (int) containing the number of clusters.
            iterations: (int) containing the maximum number of iterations.
        Returns:
            C: (numpy.ndarray) containing the centroid means for each cluster.
            clss: (numpy.ndarray) containing the index of the cluster in C
                  that each data point belongs to.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    # Get the number of data points and dimensions    
    n, d = X.shape
    # Initialize the centroids using the initialize function
    C = initialize(X, k)
    # Perform k-means clustering for maximum of iterations
    for i in range(iterations):
        # Make a copy of the centroids
        C_copy = np.copy(C)
        # Calculate the distance between eah data point and each centroid
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        # Assign each data point to the nearest centroid
        clss = np.argmin(distances, axis=0)
        for j in range(k):
            # If a centroid has no data points assigned to it reinitialize
            if len(X[clss == j]) == 0:
                C[j] = initialize(X, 1)[0]
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        # Check if the centroids have converged
        distances = np.linalg.norm(C_copy - C, axis=1)
        if np.all(distances == 0):
            break
    # Return the centroids and the cluster assignments
    return C, clss

def initialize(X, k):
    """ initializes cluster centroids for K-means
        X: np.ndarray (n, d) dataset to cluster
          n: number of data points
          d: number of dimensions
        k: positive int, number of clusters
        Returns: np.ndarray (k, d) initialized centroids for each cluster,"""
    # Check if X is a numpy ndarray with 2 dimensions
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Check if k is a positive integer
    if not isinstance(k, int) or k <= 0:
        return None

    # Get the number of data points and dimensions
    n, d = X.shape

    # Calculate the minimum and maximum values of X along each dimension
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)

    # Generate a numpy ndarray of random values between the minimum and
    # maximum values of X
    centroids = np.random.uniform(low, high, (k, d))
    return centroids