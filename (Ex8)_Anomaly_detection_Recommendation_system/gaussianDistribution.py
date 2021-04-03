import numpy as np
from scipy.stats import multivariate_normal


# TODO: Compute the mean and variance of input data set
# Return: mu, var
def estimate_gaussian_para(X):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=0)
    return mu, var


# TODO: Compute the Multivariate (Normal) Gaussian distribution (MGD) values
# Return result of MGD (normal distribution) based on their initial "mu" & "sigma" of each data example
# sigma: variance set of each features
def multivariate_gaussian(X, mu, sigma):
    p = multivariate_normal.pdf(X, mu, np.diag(sigma))
    return p

