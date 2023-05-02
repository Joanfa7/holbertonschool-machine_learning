#!/usr/bin/env python3
''' F1 score '''

import numpy as np


def f1_score(confusion):
    ''' calculates the F1 score of a confusion matrix
        confusion: np.ndarray confusion matrix (classes, classes)
        Returns: np.ndarray(classes,) with F1 score of each class
    '''
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # F1 = 2 * (precision * recall) / (precision + recall)
    TP = np.diagonal(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1
