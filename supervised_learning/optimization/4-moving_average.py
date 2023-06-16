#!/usr/bin/env python3
''' moving average '''

import tensorflow as tf


def moving_average(data, beta):
    ''' calculates the weighted moving average of a data set '''
    v = 0
    avg = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        avg.append(v / (1 - beta ** (i + 1)))
    return avg
