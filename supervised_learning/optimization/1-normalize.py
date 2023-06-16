#!/usr/bin/env python3
''' Normalization Constants '''

import numpy as np


def normalize(X, m, s):
    ''' Normalizes (standardizes) a matrix '''
    return (X - m) / s