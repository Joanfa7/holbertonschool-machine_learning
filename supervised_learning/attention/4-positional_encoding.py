#!/usr/bin/env python3
''' Positional Encoding '''

import numpy as np


def positional_encoding(max_seq_len, dm):
    ''' calculates the positional encoding for a transformer '''
    PE = np.zeros((max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / dm)))
            PE[pos, i + 1] = np.cos(pos / (10000 ** (i / dm)))
    return PE
