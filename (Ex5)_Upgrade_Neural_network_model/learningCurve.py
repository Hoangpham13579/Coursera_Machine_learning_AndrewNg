import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCost import linearRegCostFunction


# TODO: Compute train and cross validation errors between predicting and true "y" values (linear regression)
# Return ERROR of train and ERROR of cross validation (cv) data set
def learningCurve(X, y, Xvali, yvali, lambdas):
    # Initial parameters & add bias to X
    m = X.shape[0]  # X: (m, n+1); y: (m, 1)
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    # Find train error and cv error of each train and cv example
    for i in range(1, m):
        # "i": number of example trained in each situation
        X_train = X[:i, :]  # (i, n+1)
        y_train = y[:i, 0].reshape(i, 1)  # (i, 1)
        # Train theta
        theta_train = trainLinearReg(X_train, y_train, lambdas)
        # Compute train and cv error between predicting val "y" and truth val "y"
        error_train[i], _ = linearRegCostFunction(theta_train, X_train, y_train, lambdas)
        error_val[i], _ = linearRegCostFunction(theta_train, Xvali, yvali, lambdas)

    return error_train, error_val

