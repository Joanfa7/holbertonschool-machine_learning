#!/env/bin/env python3
""" performs elemetn-wise addition, subtraciton, multiplication, and division"""


def np_elementwise(mat1, mat2):
    sum_matrices = mat1 + mat2
    sub_matrices = mat1 - mat2
    mult_matrices = mat1 * mat2
    div_matrices = mat1 / mat2

    return(sum_matrices, sub_matrices, mult_matrices, div_matrices)


