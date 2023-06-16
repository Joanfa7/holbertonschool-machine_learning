#!/usr/bin/env python3
''' updates a variable using the gradient descent with momentum '''


def update_variables_momentum(alpha, beta1, var, grad, v):
    ''' updates a variable using the gradient descent with momentum '''
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
