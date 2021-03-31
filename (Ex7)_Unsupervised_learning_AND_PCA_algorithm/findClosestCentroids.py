import numpy as np


# TODO: Find the set of closest examples of each centroid
# Return: set size m in which each value is the index of closest centroid (in a set of initial centroids)
def find_closest_centroid(X, centroids):
    # X: (m, n); centroids: (K, n)
    # Initialize values
    idx = np.zeros((X.shape[0], 1))
    K = centroids.shape[0]
    temp = np.zeros((1, K))

    # Find idx of centroid closest to each data point
    for i in range(X.shape[0]):
        idx[i] = np.argmin(np.sqrt(np.sum((X[i, :] - centroids) ** 2, axis=1))) + 1

    return idx

