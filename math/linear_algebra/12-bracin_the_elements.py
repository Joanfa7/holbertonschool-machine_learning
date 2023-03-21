#!/usr/bin/env python3
""" performs elemetn-wise addition, subtraciton, multiplication, and division"""


def np_elementwise(mat1, mat2):
    ''' add, sub, mul and div matrices '''
    add = (mat1 + mat2)
    sub = (mat1 - mat2)
    mul = (mat1 * mat2)
    div = (mat1 / mat2)
    return add, sub, mul, div
