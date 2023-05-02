#!/usr/bin/env python3
''' calculates the specificity for each class in a confusion matrix '''

import numpy as np


def specificity(confusion):
    ''' calculates the specificity for each class in a confusion matrix
        confusion: np.ndarray confusion matrix (classes, classes)
        Returns: np.ndarray(classes,) of specificity of each class
    '''
    true_pos = np.diagonal(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    false_neg = np.sum(confusion, axis=1) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)
    return true_neg / (true_neg + false_pos)
