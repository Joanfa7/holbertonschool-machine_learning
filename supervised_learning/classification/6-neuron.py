#!/usr/bin/env python3=
"""
Script to create A Neuron with private instance
"""

import numpy as np


class Neuron():
    """Class Neuron"""

    def __init__(self, nx):
        """
        Args:
            nx: Type int the number of n inputs features into the ANW
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Returns: private instance weight
        """
        return self.__W

    @property
    def b(self):
        """
        Returns: private instance bias
        """
        return self.__b

    @property
    def A(self):
        """
        Returns: private instance output
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Args:
            X: Type np array with shape (nx, m) that contains the input data
        Returns: private instance output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            Y: Type np array with shape (1, m) that contains the correct
            labels for the input data
            A: Type np array with shape (1, m) containing the activated
            output of the neuron for each example
        Returns: the cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y)
                               * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        Args:
            X: Type np array with shape (nx, m) that contains the input data
            Y: Type np array with shape (1, m) that contains the correct
            labels for the input data
        Returns: the neuron’s prediction and the cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        Args:
            X: Type np array with shape (nx, m) that contains the input data
            Y: Type np array with shape (1, m) that contains the correct
            labels for the input data
            A: Type np array with shape (1, m) containing the activated
            output of the neuron for each example
            alpha: Type float the learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        Args:
            X: Type np array with shape (nx, m) that contains the input data
            Y: Type np array with shape (1, m) that contains the correct
            labels for the input data
            iterations: Type int the number of iterations to train over
            alpha: Type float the learning rate
        Returns: the evaluation of the training data after iterations of
        training have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
