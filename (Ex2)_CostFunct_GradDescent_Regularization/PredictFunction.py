import numpy as np
from Activation_function import sigmoid


# TODO: Prediction for logistic regression
# Return: Prob getting 1 & predicting val (0 or 1)
def logistic_prediction(theta, X):
    # X: (m, n); theta: (n, 1)
    m = X.shape[0]  # Number of examples

    # Guarantee the size of parameter
    p = np.zeros((m, 1))

    # Probability of getting value 1 -> Make prediction by round to decimal
    p = np.round(sigmoid(np.dot(X, theta)))

    return p
