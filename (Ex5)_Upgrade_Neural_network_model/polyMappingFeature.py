import numpy as np


# TODO: Mapping features from linear to polynomial regressions
# Return: New X with "p" polynomial features
# X_poly[i, :] = [X[i], X[i]**2, X[i]**3, ..., X[i]**p]
def poly_features(X, p):      # X: (m,)
    # Initialize parameter
    X_poly = np.zeros((X.shape[0], p))

    # Mapping features
    for i in range(p):
        X_poly[:, i] = X**(i+1)
    return X_poly  # (m, p)


# TODO: Normalize the features in X using Z-score normalization
# Return: Normalize version of X where mean of feature = 0; std of features = 1
def feature_normalize(X):  # X: (m, p)
    # Compute norm, mean, std of each feature in array
    norm = np.linalg.norm(X, axis=0)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)

    # Normalize feature of X
    # X_normal = X / norm
    X_normal = (X-mean) / std
    return X_normal, mean, std

