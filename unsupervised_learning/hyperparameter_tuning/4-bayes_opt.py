#!/usr/bin/env python3
""" Bayesian Optimization """

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ Performs Bayesian Optimization on a noiseless 1D Gaussian process """

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """ Class constructor """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ Calculates the next best sample location """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            Y_s = np.min(self.gp.Y)
            imp = Y_s - mu - self.xsi
        else:
            Y_s = np.max(self.gp.Y)
            imp = mu - Y_s - self.xsi
        Z = imp / sigma
        for idx in range(sigma.shape[0]):
            if sigma[idx] > 0:
                Z[idx] = imp[idx] / sigma[idx]
            else:
                Z[idx] = 0
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
