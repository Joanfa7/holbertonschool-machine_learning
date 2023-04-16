#!/usr/bin/env python3=
""" One Hot Decode """

import numpy as np


def one_hot_decode(one_hot):
    """ One Hot Decode """
    if not isinstance(one_hot, np.ndarray) or len(one_hot) == 2:
        return np.argmax(one_hot, axis=0)
    return None