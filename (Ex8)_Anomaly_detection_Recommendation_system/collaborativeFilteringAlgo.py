import numpy as np


# TODO: Compute the cost and the gradient of collaborative filtering learning algorithm
# Return: the cost value & gradient of cost in terms of X and theta
def cofi_learning_algo(params, Y, R, num_users, num_movies, num_features, lambda_r):
    # Extract value of X & y from params
    # order="F" makes the reshape() faster
    # "Y": (num_movies * num_users), "R": (num_movies, num_users)
    X = np.reshape(params[:num_movies*num_features], newshape=(num_movies, num_features), order="F")
    Theta = np.reshape(params[num_movies*num_features:], newshape=(num_users, num_features), order="F")

    # Compute the cost function (with regularization)
    C = np.subtract(np.dot(X, Theta.T), Y)**2
    J = np.sum(np.sum(R*C))/2 + ((lambda_r/2)*np.sum(np.sum(X**2))) + ((lambda_r/2)*np.sum(np.sum(Theta**2)))

    # Compute gradient of cost in terms of X and theta
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # Compute the gradient in terms of X
    for i in range(num_movies):
        idx = np.argwhere(R[i, :] == 1).T[0]
        X_grad[i, :] = np.dot(np.subtract(np.dot(X[i, :], Theta[idx, :].T), Y[i, idx]), Theta[idx, :]) +\
                       (lambda_r * X[i, :])

    # Compute the gradient in terms of Theta
    for j in range(num_users):
        idx = np.argwhere(R[:, j] == 1).T[0]
        Theta_grad[j, :] = np.dot(np.subtract(np.dot(X[idx, :], Theta[j, :].T), Y[idx, j]).T, X[idx, :]) +\
                           (lambda_r * Theta[j, :])

    grad = np.hstack((X_grad.ravel(order="F"), Theta_grad.ravel(order="F")))
    return J, grad


# TODO: Normalize the rating dataset
# Return: normalized rating dataset
def normalize_rating(Y, R):
    # Y: (num_movies, num_users)
    Y_mean = np.zeros((Y.shape[0], 1))
    Y_norm = np.zeros(Y.shape)

    for i in range(Y.shape[0]):
        idx = np.argwhere(R[i, :] == 1).T[0]
        Y_mean[i] = np.mean(Y[i, idx])
        Y_norm[i, idx] = np.subtract(Y[i, idx], Y_mean[i]) / (np.std(Y[i, idx]))
    return Y_norm, Y_mean
