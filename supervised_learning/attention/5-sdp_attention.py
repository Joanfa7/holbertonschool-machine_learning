#!/usr/bin/env python3

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ sdp_attention - calculates the scaled dot product attention """
    Q = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled = Q / tf.math.sqrt(dk)
    if mask is not None:
        scaled += (mask * -1e9)
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
