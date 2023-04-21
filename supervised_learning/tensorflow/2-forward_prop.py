#!/usr/bin/env python3
''' Forward Propagation'''

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward Propagation"""
    for i in range(len(layer_sizes)):
        if i == 0:
            y_pred = create_layer(x, layer_sizes[i], activations[i])
        else:
            y_pred = create_layer(y_pred, layer_sizes[i], activations[i])
    return y_pred
