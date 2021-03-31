import numpy as np
from Activation_function import sigmoid
from FeatureMapping import features_mapping


# TODO Cost function for linear regression for classification prob
def cost_function_linear(theta, X, y):
    # X: (m, n);  theta: (n, 1);  y: (m, 1)
    (m, n) = X.shape

    # Cost function
    J = 0
    h = np.matmul(X, theta)  # h: (m, 1)
    square = (h-y)**2
    J = 1/(2*m) * sum(square)

    return J


# TODO Cost function for logistic regression for classification prob
def cost_function_logistic(theta, X, y):
    # m: Number of training examples; n: Number of features
    # X: (m, n);  theta: (n, 1);  y: (m, 1)
    J = 0
    m, n = X.shape
    alpha = 0.0001
    grad = np.zeros(theta.shape)

    # (NOTE) Theta DIM must be (n, 1) (2D array))
    theta = np.reshape(theta, (n, 1))

    # Cost function
    h = sigmoid(np.dot(X, theta))  # h: (m, 1)
    J = (-1/m) * np.sum(np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))

    # Derivative of J (gradient descent)
    grad = (1/m) * (np.dot(X.T, (h-y)))

    return J, grad


# TODO Cost applied Regularization for classification logistic regression
# PURPOSE: Apply regularization for solving "Overfitting" problem
def cost_logistic_regularization(theta, X, y, lambdas):
    # X: (m, n+1);  theta: (n+1,)_1D array;  y: (m, 1), h: (m, 1)
    # Initial values
    theta = theta[:, np.newaxis]  # Increase 1D to 2D theta
    J = 0
    m = X.shape[0]
    grad = np.zeros((X.shape[1], 1))

    # Cost function apply regularization
    h = sigmoid(np.dot(X, theta))
    term = np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))
    J = ((-1/m) * term) + (lambdas/(2*m)) * np.sum(theta[1:]**2)

    # Derivative of cost function (grad: (n+1, 1))
    grad = (1/m) * (np.dot(X.T, (h-y))) + (lambdas/m) * theta
    grad[0, 0] = (1/m) * np.dot(X.T[0, :], (h-y))

    return J, grad


# (NOTE) np.hstack(), np.vstack(): concatenate 2D array (horizontal & vertical)
# (NOTE) np.matmul() OR np.dot() return 2D array (If input 2D array)
# (NOTE) np.multiply() return 1D array (Even if input 2D array)

