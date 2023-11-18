#!/usr/bin/env python3
""" K-means """


import sklearn.cluster

def kmeans(X, k):
    """ performs K-means on a dataset
    Arg:
        - X: np.ndarray shape(n, d) dataset
            - n: number of data points
            - d: number of dimensions
        - k: positive int number of clusters
    Returns: C, clss
        - C: np.ndarray shape(k, d) centroid means 
        for each cluster
        - clss: np.ndarray shape(n,) index of cluster in C that each data point belongs to
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
