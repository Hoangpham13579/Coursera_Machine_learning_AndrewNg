import numpy as np
import scipy.optimize as opt
from lrCostFunction import cost_logistic_regularization


# TODO: Train multiple logistic classifiers and return all classifiers in a matrix all_theta
# all_theta: a matrix where i-th row is a trained logistic theta vector for the i-th labels (predicted labels)
def oneVsAll(X, y, num_labels, lambdas):
    # y: Predicted label vec for each example (Ex: [1,3,5,0,6,2,...])
    # X: (m, n), num_labels: 10; y: (n, 1)
    (m, n) = X.shape

    # Confirm the dimension of all_theta & Add bias col to X
    all_theta = np.zeros((num_labels, n+1))  # all_theta: (num_labels, n+1)
    X = np.hstack((np.ones((m, 1)), X))  # X: (m, n+1)

    # Train theta for each label
    for i in range(num_labels):
        initial_theta = np.zeros(n + 1)
        # (WAY 1) Using opt.fmin_tnc() function
        # theta = opt.fmin_tnc(func=cost_logistic_regularization,
        #                      # (NOTE) X0 must be 1D array
        #                      x0=initial_theta, args=(X, (y == i+1), lambdas))

        # (WAY 2) Using opt.minimize() function (Recommend)
        result = opt.minimize(cost_logistic_regularization, initial_theta,
                              # (NOTE) initial_theta must be 1D array
                              args=(X, y == i+1, lambdas), method="TNC",
                              # "jac=True" used when cost_func return (, grad)
                              jac=True, options={"maxiter": 50})

        all_theta[i, :] = result["x"]
    return all_theta

