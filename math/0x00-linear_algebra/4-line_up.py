#!/usr/bin/env python3
""" Function that adds tww arrays element-wise """


def add_arrays(arr1, arr2):
    """ Funciton that evaluates if the arrays are the same
    lengh if they are prossed to add theri elements"""
    if len(arr1) != len(arr2):
        return [None]
    else:
        new_array = []
        for idx in range(0, len(arr1)):
            new_array.append(arr1[idx] + arr2[idx])
        return new_array
