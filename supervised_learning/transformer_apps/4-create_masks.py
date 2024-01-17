#!/usr/bin/env python3
''' Module for creating masks to be used in the transformer '''
import tensorflow as tf


def create_masks(inputs, target):
    ''' creates all masks for training/validation '''
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = tf.expand_dims(encoder_mask, 1)
    encoder_mask = tf.expand_dims(encoder_mask, 1)
    
    look_ahead_mask = tf.cast(tf.math.greater(tf.range(tf.shape(target)[1]), tf.range(tf.shape(target)[1])[:, tf.newaxis]), tf.float32)
    look_ahead_mask = tf.expand_dims(look_ahead_mask, 1)

    decoder_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_mask = tf.expand_dims(decoder_mask, 1)
    decoder_mask = tf.expand_dims(decoder_mask, 1)

    combined_mask = tf.maximum(decoder_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
