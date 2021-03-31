import numpy as np
from linearRegCost import linearRegCostFunction
import scipy.optimize as opt


# TODO: Train linear regression using (X, y) and regularization using lambda
def trainLinearReg(X, y, lambdas):
    # X: (m, n+1)
    # Initialize parameters & Add bias to X
    m = X.shape[0]
    initial_theta = np.zeros(X.shape[1])  # (n+1, )

    # Train model linear regression
    result = opt.minimize(linearRegCostFunction, initial_theta,
                          args=(X, y, lambdas), method="TNC",
                          jac=True, options={"maxiter": 50})
    train_theta = result["x"]

    return train_theta.reshape(X.shape[1], 1)



# (NOTE) method="TNC" is used for optimizing linear regression cost function
# (NOTE) method="L-BFGS-B" is used for optimizing logistic regression (non-linear) cost function
