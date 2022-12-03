#!/usr/bin/env python3
""" function that adds two matrices elements element-wise"""


def add_matrices2D(mat1, mat2):
    """ add 2 matixes"""
    if len(mat1[0]) != len(mat2[0]):
        return None
    else:
        res = []
        for i in range(len(mat1)):
            row= []
            for j in range(len(mat1[0])):
                row.append(mat1[i][j]+mat2[i][j])
            res.append(row)
        return res