import numpy as np


# TODO: Normalizing features before applying PCA
# Output: normalized_X, mean, std
def feature_normalize(X):
    mean = np.mean(X, axis=0)
    X_norm = np.subtract(X, mean)

    std = np.std(X_norm, axis=0, ddof=1)
    X_norm = X_norm/std
    return X_norm, mean, std


