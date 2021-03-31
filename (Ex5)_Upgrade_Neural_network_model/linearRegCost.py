import numpy as np


# TODO: Return cost value for linear regression applied regularization
def linearRegCostFunction(theta, X, y, lambdas):
    # X: (m, n+1); theta: (n+1, 1);  y: (m, 1), h: (m, 1)
    # Initialize parameter & add bias to X
    m = X.shape[0]
    theta = theta.reshape(X.shape[1], 1)  # theta: (n+1, 1)

    # Cost function applying regularization
    h = np.dot(X, theta)
    square = (h-y)**2
    J = (1/(2*m)) * np.sum(square) + (lambdas/(2*m)) * np.sum(theta[1:, :]**2)

    # Compute regularized gradient
    grad = (1/m) * np.dot(X.T, (h-y)) + (lambdas/m) * theta
    grad[0, 0] = (1/m) * np.dot(X.T[0, :], (h-y))

    return J, grad

