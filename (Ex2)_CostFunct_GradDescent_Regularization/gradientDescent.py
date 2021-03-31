import numpy as np
from CostFunction import cost_function_linear
from Activation_function import sigmoid


# TODO Gradient descent for linear regression
def gradient_descent_linear(X, y, theta, alpha, num_iters):
    # X:(m, n); y:(m, 1); theta:(n, 1); alpha: learning rate

    n = X.shape[1]
    m = len(y)  # Num of examples
    J_history = np.zeros(num_iters, 1)  # Cost after each gradient descent

    # Gradient descent
    for iter in range(num_iters):
        # Derivative of cost function
        h = np.matmul(X, theta)  # h: (m,1)
        derivative = 1/m * np.matmul(X.T, h-y)

        # Update theta
        theta = theta - alpha*derivative
        J_history[iter] = cost_function_linear(X, y, theta)

    return J_history


# TODO Gradient descent for logistic regression
def gradient_logistic(X, y, theta):
    # X: (m, n);  theta: (n, 1);  y: (m, 1)
    m = X.shape[0]
    # Activation function
    h = sigmoid(np.dot(X, theta))  # h: (m, 1)

    # Gradient descent
    grad = (1/m) * (np.dot(X.T, (h-y)))
    return grad


# TODO: Gradient descent for logistic regression apply regularization
def gradient_logistic_regularization(theta, X, y, lambdas):  # (NOT DONE)
    # X: (m, n+1);  theta: (n+1, 1);  y: (m, 1), h: (m, 1)
    # Initial values
    theta = np.array(theta)  # Confirm 2D array
    J = 0;
    m = X.shape[0]
    grad = np.zeros((X.shape[1], 1))

    # Prepare for regularization
    theta_reg = theta[1:theta.shape[0]]  # theta_reg: 1D array
    theta_reg = np.insert(theta_reg, 0, 0)
    theta_reg = np.reshape(theta_reg, (X.shape[1], 1))  # Confirm 2D array

    # Derivative of cost function (grad: (n+1, 1))
    h = sigmoid(np.dot(X, theta))
    grad = (1 / m) * (np.dot(X.T, (h - y))) + (lambdas / m) * theta_reg

    return grad
