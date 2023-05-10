#!/usr/bin/env python3
''' Function that saves a model’s configuration in JSON format '''

import tensorflow.keras as K


def save_config(network, filename):
    ''' Function that saves a model’s configuration in JSON format '''
    json_model = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_model)
    return None


def load_config(filename):
    ''' Function that loads a model with a specific configuration '''
    with open(filename, 'r') as json_file:
        json_model = json_file.read()
    return K.models.model_from_json(json_model)
