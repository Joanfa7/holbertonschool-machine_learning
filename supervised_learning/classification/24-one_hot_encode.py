#!/usr/bin/env python3
""" One Hot Encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ One Hot Encode """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None
    one_hot = np.zeros((classes, len(Y)))
    one_hot[Y, np.arange(len(Y))] = 1
    return one_hot
