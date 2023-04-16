#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork """

    def __init__(self, nx, layers):
        """ Class constructor """
        if isinstance(layers, int):
            layers = [layers]

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2 / nx)
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        """ Getter method for L """
        return self.__L

    @property
    def cache(self):
        """ Getter method for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter method for weights """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for i in range(self.__L):
            z = np.matmul(self.__weights["W{}".format(i + 1)],
                          self.__cache["A{}".format(i)]) + \
                self.__weights["b{}".format(i + 1)]
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-z))
        return self.__cache["A{}".format(i + 1)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        A, _ = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        m = Y.shape[1]
        dz = cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            dw = 1 / m * np.matmul(dz, cache["A{}".format(i - 1)].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            dz = np.matmul(self.__weights["W{}".format(i)].T, dz) * \
                (cache["A{}".format(i - 1)] * (1 - cache["A{}".format(i - 1)]))
            self.__weights["W{}".format(i)] -= alpha * dw
            self.__weights["b{}".format(i)] -= alpha * db

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """ Trains the deep neural network """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        iters = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and i % step == 0:
                print(
                    "Cost after {} iterations: {}".format(
                        i, self.cost(
                            Y, A)))
                costs.append(self.cost(Y, A))
                iters.append(i)
        if graph:
            plt.plot(iters, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
