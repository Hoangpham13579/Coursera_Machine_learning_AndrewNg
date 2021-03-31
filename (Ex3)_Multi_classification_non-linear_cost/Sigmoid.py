import numpy as np


# TODO Logistic function (Sigmoid function)
def sigmoid(z):
    # Guarantee size of sigmoid value
    sig = np.zeros(z.shape)

    # Sigmoid function
    sig = 1 / (1 + np.exp(-z))

    return sig