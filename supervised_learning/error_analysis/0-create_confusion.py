#!/usr/bin/env python3

"""Function that creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix"""
    return np.matmul(labels.T, logits)
