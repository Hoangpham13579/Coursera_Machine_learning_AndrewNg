import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCost import linearRegCostFunction


# TODO
# Input: training set (X, y) & Validation set (Xval, yval)
# Output: Lambda set corresponding with train and validation error
def validation_curve(X, y, Xvali, yvali):
    # Selected set of lambdas
    lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # Generate initial parameters
    m = X.shape[0]  # X: (m, n+1); y: (m, 1)
    error_train = np.zeros((len(lambdas), 1))
    error_val = np.zeros((len(lambdas), 1))

    # Compute error of train and validation error based on each value of lambdas
    for i in range(len(lambdas)):  # 0-9
        lambda_r = lambdas[i]
        theta = trainLinearReg(X, y, lambda_r)
        error_train[i], _ = linearRegCostFunction(theta, X, y, lambda_r)
        error_val[i], _ = linearRegCostFunction(theta, Xvali, yvali, lambda_r)

    return lambdas, error_train, error_val


