#!/usr/bin/env python3
""" Function that concatenates two matrices along a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenat a matrix"""
    contcat_matrix = np.concatenate((mat1, mat2), axis=axis)

    return contcat_matrix
