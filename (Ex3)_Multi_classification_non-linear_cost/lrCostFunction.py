import numpy as np
from Sigmoid import sigmoid


# TODO Cost applied Regularization for classification logistic regression
# PURPOSE: Apply regularization for solving "Overfitting" problem
def cost_logistic_regularization(theta, X, y, lambdas):
    # X: (m, n+1);  theta: (n+1, 1);  y: (m, 1), h: (m, 1)
    # Initial values
    theta = theta[:, np.newaxis]  # Increase 1D to 2D array
    J = 0; m = X.shape[0]
    grad = np.zeros((X.shape[1], 1))

    # Cost function apply regularization
    h = sigmoid(np.dot(X, theta))
    term = np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))
    J = ((-1/m) * term) + (lambdas/(2*m)) * np.sum(theta[1:]**2)

    # Derivative of cost function (grad: (n+1, 1))
    grad = (1/m) * (np.dot(X.T, (h-y))) + (lambdas/m) * theta
    grad[0, 0] = (1/m) * np.dot(X.T[0, :], (h-y))

    return float(J), grad
