import numpy as np


# (NOTE) Sigmoid function used for logistic regression (non-linear regression)
# TODO Logistic function (Sigmoid function)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# TODO return gradient (derivative) of sigmoid function
def sigmoid_gradient(z):
    return sigmoid(z) * (1-sigmoid(z))

