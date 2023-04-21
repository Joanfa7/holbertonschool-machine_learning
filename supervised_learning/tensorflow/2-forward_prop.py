#!/usr/bin/env python3
"""Create Layer"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_size=[], activations=[]):
    """Creates the forward propagation graph for the neural network
    Args:
        x: is the placeholder for the input data
        layer_size: is a list containing the number of nodes in each layer
        of the network
        activations: is a list containing the activation functions for each
        layer of the network
    Returns:
        the prediction of the network in tensor form
    """
    for i in range(len(layer_size)):
        layer = create_layer(x, layer_size[i], activations[i])
    return layer
